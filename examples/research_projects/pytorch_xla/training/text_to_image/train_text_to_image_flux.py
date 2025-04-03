import functools
import argparse
import os
import random
import time
from pathlib import Path

import copy
import datasets
from datasets import concatenate_datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from huggingface_hub import create_repo, upload_folder
from torchvision import transforms
from torchvision.transforms.functional import crop
from transformers import CLIPTextModel, CLIPTokenizer, PretrainedConfig, T5EncoderModel, T5TokenizerFast
from torch_xla.experimental.custom_kernel import FlashAttention
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.training_utils import compute_snr
from diffusers.utils import is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card

if is_wandb_available():
    pass

PROFILE_DIR = os.environ.get("PROFILE_DIR", None)
CACHE_DIR = os.environ.get("CACHE_DIR", None)
if CACHE_DIR:
    xr.initialize_cache(CACHE_DIR, readonly=False)
xr.use_spmd()
DATASET_NAME_MAPPING = {
    "lambdalabs/naruto-blip-captions": ("image", "text"),
}
PORT = 9012


def save_model_card(
    args,
    repo_id: str,
    repo_folder: str = None,
):
    model_description = f"""
# Text-to-image finetuning - {repo_id}

This pipeline was finetuned from **{args.pretrained_model_name_or_path}** on the **{args.dataset_name}** dataset. \n

## Pipeline usage

You can use the pipeline like so:

```python
import torch
import os
import sys
import  numpy as np

import torch_xla.core.xla_model as xm
from time import time
from typing import Tuple
from diffusers import StableDiffusionPipeline

def main(args):
    device = xm.xla_device()
    model_path = <output_dir>
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    )
    pipe.to(device)
    prompt = ["A naruto with green eyes and red legs."]
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    image.save("naruto.png")

if __name__ == '__main__':
    main()
```

## Training info

These are the key hyperparameters used during training:

* Steps: {args.max_train_steps}
* Learning rate: {args.learning_rate}
* Batch size: {args.train_batch_size}
* Image resolution: {args.resolution}
* Mixed-precision: {args.mixed_precision}

"""

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=args.pretrained_model_name_or_path,
        model_description=model_description,
        inference=True,
    )

    tags = ["stable-diffusion", "stable-diffusion-diffusers", "text-to-image", "diffusers", "diffusers-training"]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


class TrainSD:
    def __init__(
        self,
        weight_dtype,
        device,
        noise_scheduler,
        vae_scale_factor,
        transformer,
        optimizer,
        dataloader,
        args,
    ):
        self.weight_dtype = weight_dtype
        self.device = device
        self.noise_scheduler = noise_scheduler
        self.transformer = transformer
        self.optimizer = optimizer
        self.args = args
        self.mesh = xs.get_global_mesh()
        self.dataloader = iter(dataloader)
        self.global_step = 0
        self.noise_scheduler_copy = copy.deepcopy(noise_scheduler)
        self.vae_scale_factor = vae_scale_factor

    def run_optimizer(self):
        self.optimizer.step()

    def start_training(self):
        dataloader_exception = False
        measure_start_step = self.args.measure_start_step
        print("meaure_start_step: ", measure_start_step)
        print("max_train_steps: ", self.args.max_train_steps)
        assert measure_start_step < self.args.max_train_steps
        total_time = 0
        for step in range(0, self.args.max_train_steps):
            print("step: ", step)
            batch = next(self.dataloader)
            if step == measure_start_step:
                if PROFILE_DIR is not None:
                    xm.wait_device_ops()
                    xp.trace_detached(f"localhost:{PORT}", PROFILE_DIR, duration_ms=args.profile_duration)
                last_time = time.time()
            loss = self.step_fn(batch)
            self.global_step += 1

            def print_loss_closure(step, loss):
                print(f"Step: {step}, Loss: {loss}")

            if self.args.print_loss:
                xm.add_step_closure(
                    print_loss_closure,
                    args=(
                        self.global_step,
                        loss,
                    ),
                )
        xm.mark_step()
        if not dataloader_exception:
            xm.wait_device_ops()
            total_time = time.time() - last_time
            print(f"Average step time: {total_time/(self.args.max_train_steps-measure_start_step)}")
        else:
            print("dataloader exception happen, skip result")
            return
    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.noise_scheduler_copy.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler_copy.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    def step_fn(
        self,
        batch
    ):
        with xp.Trace("model.forward"):
            self.optimizer.zero_grad()
            prompt_embeds = batch["prompt_embeds"]
            pooled_prompt_embeds = batch["pooled_prompt_embeds"]
            text_ids = batch["text_ids"]
            model_input = batch["model_input"]

            latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                model_input.shape[0],
                model_input.shape[2] // 2,
                model_input.shape[3] // 2,
                self.device,
                self.weight_dtype,
            )

            noise = torch.randn_like(model_input).to(self.device, dtype=self.weight_dtype)
            bsz = model_input.shape[0]

            # Sample a random timestep for each image
            # for weighting schemes where we sample timesteps non-uniformly
            u = compute_density_for_timestep_sampling(
                weighting_scheme=args.weighting_scheme,
                batch_size=bsz,
                logit_mean=args.logit_mean,
                logit_std=args.logit_std,
                mode_scale=args.mode_scale,
            )
            indices = (u * self.noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = self.noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

            # Add noise according to flow matching.
            # zt = (1 - texp) * x + texp * z1
            sigmas = self.get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
            noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

            packed_noisy_model_input = FluxPipeline._pack_latents(
                noisy_model_input,
                batch_size=model_input.shape[0],
                num_channels_latents=model_input.shape[1],
                height=model_input.shape[2],
                width=model_input.shape[3],
            )

            # handle guidance
            if self.transformer.config.guidance_embeds:
                guidance = torch.tensor([args.guidance_scale], device=self.device)
                guidance = guidance.expand(model_input.shape[0])
            else:
                guidance = None

            # Predict the noise residual
            model_pred = self.transformer(
                hidden_states=packed_noisy_model_input,
                # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                timestep=timesteps / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]
            # upscaling height & width as discussed in https://github.com/huggingface/diffusers/pull/9257#discussion_r1731108042
            model_pred = FluxPipeline._unpack_latents(
                model_pred,
                height=model_input.shape[2] * self.vae_scale_factor,
                width=model_input.shape[3] * self.vae_scale_factor,
                vae_scale_factor=self.vae_scale_factor,
            )

            # these weighting schemes use a uniform timestep sampling
            # and instead post-weight the loss
            weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

            # flow matching loss
            target = noise - model_input

        with xp.Trace("model.backward"):
            # Compute regular loss.
            loss = torch.mean(
                (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                1,
            )
            loss = loss.mean()
            loss.backward()
        with xp.Trace("optimizer_step"):
            self.run_optimizer()
        return loss


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--profile_duration", type=int, default=10000, help="Profile duration in ms")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the FLUX.1 dev variant is a guidance distilled model",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--loader_prefetch_size",
        type=int,
        default=1,
        help=("Number of subprocesses to use for data loading to cpu."),
    )
    parser.add_argument(
        "--loader_prefetch_factor",
        type=int,
        default=2,
        help=("Number of batches loaded in advance by each worker."),
    )
    parser.add_argument(
        "--device_prefetch_size",
        type=int,
        default=1,
        help=("Number of subprocesses to use for data loading to tpu from cpu. "),
    )
    parser.add_argument("--measure_start_step", type=int, default=10, help="Step to start profiling.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "bf16"],
        help=("Whether to use mixed precision. Bf16 requires PyTorch >= 1.10"),
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--print_loss",
        default=False,
        action="store_true",
        help=("Print loss at every step."),
    )

    args = parser.parse_args()

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def setup_optimizer(transformer, args):
    optimizer_cls = torch.optim.AdamW
    return optimizer_cls(
        transformer.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
        foreach=True,
    )

def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds

def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds

def encode_prompt(
    batch,
    text_encoders,
    tokenizers,
    caption_column,
    max_sequence_length = 512,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = batch[caption_column]
    batch_size = len(prompt)
    dtype = text_encoders[0].dtype
    device = device if device is not None else text_encoders[1].device
    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device,
    )

    text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
    text_ids = text_ids.repeat(num_images_per_prompt, 1, 1)

    return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds, "text_ids" : text_ids}

def compute_vae_encodings(batch, vae, device, dtype):
    images = batch.pop("pixel_values")
    pixel_values = torch.stack(list(images))
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)

    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
    return {"model_input": model_input}

def pixels_to_tensors(batch, device, dtype):
    images = batch.pop("pixel_values")
    pixel_values = torch.stack(list(images))
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    return {"pixel_tensor_values" : pixel_values}

def load_dataset(args):
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = datasets.load_dataset(
            args.dataset_name,
            cache_dir=args.cache_dir,
            data_dir=args.train_data_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = datasets.load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
    return dataset


def get_column_names(dataset, args):
    column_names = dataset["train"].column_names

    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )
    return image_column, caption_column


def main(args):
    args = parse_args()
    cache_path = Path("/tmp/data/compiler_cache")
    cache_path.mkdir(parents=True, exist_ok=True)
    xr.initialize_cache(str(cache_path), readonly=False)

    _ = xp.start_server(PORT)

    num_devices = xr.global_runtime_device_count()
    mesh = xs.get_1d_mesh("data")
    xs.set_global_mesh(mesh)

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
    )
    text_encoder_2 = T5EncoderModel.from_pretrained(
      args.pretrained_model_name_or_path,
      subfolder="text_encoder_2",
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )

    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
    )

    if xm.is_master_ordinal() and args.push_to_hub:
        repo_id = create_repo(
            repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
        ).repo_id

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False
    )
    tokenizer_2 = T5TokenizerFast.from_pretrained(
      args.pretrained_model_name_or_path,
      subfolder="tokenizer_2",
      revision=args.revision,
      use_fast=False
    )

    from torch_xla.distributed.fsdp.utils import apply_xla_patch_to_nn_linear

    #transformer = apply_xla_patch_to_nn_linear(transformer, xs.xla_patched_nn_linear_forward)
    transformer.enable_xla_flash_attention(partition_spec=("data", None, None, None), is_flux=True)
    FlashAttention.DEFAULT_BLOCK_SIZES = {
        "block_q": 512,
        "block_k_major": 512,
        "block_k": 512,
        "block_b": 512,
        "block_q_major_dkv": 512,
        "block_k_major_dkv": 512,
        "block_q_dkv": 512,
        "block_k_dkv": 512,
        "block_q_dq": 512,
        "block_k_dq": 768,
        "block_k_major_dq": 512,
    }
    # For mixed precision training we cast all non-trainable weights (vae,
    # non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full
    # precision is not required.
    weight_dtype = torch.float32
    if args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    device = xm.xla_device()

    # Move text_encode and vae to device and cast to weight_dtype
    text_encoder = text_encoder.to(device, dtype=weight_dtype)
    text_encoder_2 = text_encoder_2.to(device, dtype=weight_dtype)
    vae = vae.to(device, dtype=weight_dtype)
    transformer = transformer.to(device, dtype=weight_dtype)
    optimizer = setup_optimizer(transformer, args)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    transformer.train()

    dataset = load_dataset(args)
    image_column, caption_column = get_column_names(dataset, args)

    train_resize = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
    train_crop = transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution)
    train_flip = transforms.RandomHorizontalFlip(p=1.0)
    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        original_sizes = []
        all_images = []
        for image in images:
          original_sizes.append((image.height, image.width))
          image = train_resize(image)
          if args.random_flip and random.random() < 0.5:
              # flip
              image = train_flip(image)
          if args.center_crop:
              y1 = max(0, int(round((image.height - args.resolution) / 2.0)))
              x1 = max(0, int(round((image.width - args.resolution) / 2.0)))
              image = train_crop(image)
          else:
              y1, x1, h, w = train_crop.get_params(image, (args.resolution, args.resolution))
              image = crop(image, y1, x1, h, w)
          image = train_transforms(image)
          all_images.append(image)
        examples["pixel_values"] = all_images
        return examples

    train_dataset = dataset["train"]
    train_dataset.set_format("torch")
    train_dataset.set_transform(preprocess_train)

    text_encoders = [text_encoder, text_encoder_2]
    tokenizers = [tokenizer, tokenizer_2]
    compute_embeddings_fn = functools.partial(
      encode_prompt,
      text_encoders=text_encoders,
      tokenizers=tokenizers,
      caption_column=caption_column,
    )
    compute_vae_encodings_fn = functools.partial(compute_vae_encodings, vae=vae, device=device, dtype=weight_dtype)
    from datasets.fingerprint import Hasher

    new_fingerprint = Hasher.hash(args)
    new_fingerprint_two = Hasher.hash((args.pretrained_model_name_or_path, args))
    train_dataset_with_embeddings = train_dataset.map(
        compute_embeddings_fn, batched=True, new_fingerprint=new_fingerprint
    )
    train_dataset_with_tensors = train_dataset.map(
        compute_vae_encodings_fn, batched=True, new_fingerprint=new_fingerprint_two, batch_size=8
    )
    precomputed_dataset = concatenate_datasets(
        [train_dataset_with_embeddings, train_dataset_with_tensors.remove_columns(["text", "image"])], axis=1
    )
    precomputed_dataset = precomputed_dataset.with_transform(preprocess_train)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    del compute_embeddings_fn, text_encoder, text_encoder_2, vae
    del text_encoders, tokenizers
    def collate_fn(examples):
        prompt_embeds = torch.stack([torch.tensor(example["prompt_embeds"]) for example in examples]).to(dtype=weight_dtype)
        pooled_prompt_embeds = torch.stack([torch.tensor(example["pooled_prompt_embeds"]) for example in examples]).to(dtype=weight_dtype)
        text_ids = torch.stack([torch.tensor(example["text_ids"]) for example in examples]).to(dtype=weight_dtype)
        model_input = torch.stack([torch.tensor(example["model_input"]) for example in examples]).to(dtype=weight_dtype)
        return {
          "prompt_embeds": prompt_embeds,
          "pooled_prompt_embeds": pooled_prompt_embeds,
          "text_ids" : text_ids,
          "model_input" : model_input,
        }

    g = torch.Generator()
    g.manual_seed(xr.host_index())
    sampler = torch.utils.data.RandomSampler(precomputed_dataset, replacement=True, num_samples=int(1e10), generator=g)
    train_dataloader = torch.utils.data.DataLoader(
        precomputed_dataset,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        batch_size=args.train_batch_size,
        prefetch_factor=args.loader_prefetch_factor,
    )

    train_dataloader = pl.MpDeviceLoader(
        train_dataloader,
        device,
        input_sharding={
            "prompt_embeds" : xs.ShardingSpec(mesh, ("data", None, None), minibatch=True),
            "pooled_prompt_embeds" : xs.ShardingSpec(mesh, ("data", None,), minibatch=True),
            "model_input" : xs.ShardingSpec(mesh, ("data", None, None, None), minibatch=True),
            "text_ids" : xs.ShardingSpec(mesh, ("data", None, None), minibatch=True),
        },
        loader_prefetch_size=args.loader_prefetch_size,
        device_prefetch_size=args.device_prefetch_size,
    )

    num_hosts = xr.process_count()
    num_devices_per_host = num_devices // num_hosts
    if xm.is_master_ordinal():
        print("***** Running training *****")
        print(f"Instantaneous batch size per device = {args.train_batch_size // num_devices_per_host }")
        print(
            f"Total train batch size (w. parallel, distributed & accumulation) = {args.train_batch_size * num_hosts}"
        )
        print(f"  Total optimization steps = {args.max_train_steps}")

    trainer = TrainSD(
        weight_dtype=weight_dtype,
        device=device,
        noise_scheduler=noise_scheduler,
        vae_scale_factor=vae_scale_factor,
        transformer=transformer,
        optimizer=optimizer,
        dataloader=train_dataloader,
        args=args,
    )
    trainer.start_training()
    # unet = trainer.unet.to("cpu")
    
    # text_encoder = CLIPTextModel.from_pretrained(
    #     args.pretrained_model_name_or_path,
    #     subfolder="text_encoder",
    #     revision=args.revision,
    #     variant=args.variant,
    # )
    # text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
    #   args.pretrained_model_name_or_path,
    #   subfolder="text_encoder_2",
    #   revision=args.revision,
    #   variant=args.variant,
    # )
    # vae = AutoencoderKL.from_pretrained(
    #     args.pretrained_model_name_or_path,
    #     subfolder="vae",
    #     revision=args.revision,
    #     variant=args.variant,
    # )

    # pipeline = StableDiffusionXLPipeline.from_pretrained(
    #     args.pretrained_model_name_or_path,
    #     text_encoder=text_encoder,
    #     vae=vae,
    #     unet=unet,
    #     revision=args.revision,
    #     variant=args.variant,
    # )
    # pipeline.save_pretrained(args.output_dir)

    # if xm.is_master_ordinal() and args.push_to_hub:
    #     save_model_card(args, repo_id, repo_folder=args.output_dir)
    #     upload_folder(
    #         repo_id=repo_id,
    #         folder_path=args.output_dir,
    #         commit_message="End of training",
    #         ignore_patterns=["step_*", "epoch_*"],
    #     )


if __name__ == "__main__":
    args = parse_args()
    main(args)
