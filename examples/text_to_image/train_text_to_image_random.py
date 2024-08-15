
from typing import Tuple
import sys
import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import time
import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
# from accelerate import Accelerator
from datasets import load_dataset
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

import torch_xla.debug.metrics as met
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_backend
import torch_xla.debug.profiler as xp
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.distributed as dist
import torch_xla.distributed.xla_backend

PROFILE_DIR='gs://bbahl/sd2/'

xr.initialize_cache('/tmp/xla_cache/', readonly=False)

torch_xla.experimental.eager_mode(True)
xr.use_spmd()

class TrainSD():

    def __init__(self, vae, weight_dtype, device, noise_scheduler, unet, optimizer, text_encoder, args):
        self.vae = vae
        self.weight_dtype = weight_dtype
        self.device = device
        self.noise_scheduler = noise_scheduler
        self.unet = unet
        self.optimizer = optimizer
        self.text_encoder = text_encoder
        self.args = args
        self.mesh = xs.get_global_mesh()
        
        self.compiled_step_fn = torch_xla.experimental.compile(
            self.step_fn
        )
        self.global_step = 0
        self.pixel_values = torch.randn((self.args.train_batch_size, 3, args.resolution, args.resolution), dtype=torch.bfloat16, device=self.device)
        self.input_ids = torch.randint(0, 150, (self.args.train_batch_size,77), dtype=torch.int64,device=self.device)
        xs.mark_sharding(self.pixel_values, self.mesh, ('batch', None, None, None))
        xs.mark_sharding(self.input_ids, self.mesh, ('batch', None))

    def run_optimizer(self):
        self.optimizer.step()
    
    def train_loop_fn(self, epoch):
        last_time = time.time()
        for step in range(self.args.max_train_steps):
            if step == 2:
                xp.trace_detached('localhost:9012', PROFILE_DIR, duration_ms=8000)
            self.compiled_step_fn(self.pixel_values, self.input_ids)
            xm.mark_step()
            print(f"step: {step}, step_time: {time.time() - last_time}")
            last_time = time.time()
            self.global_step += 1
    
    def start_training(self):
        for epoch in range(0, self.args.num_train_epochs):
            self.train_loop_fn(epoch)
        xm.wait_device_ops()
            

    def train_update(self, step, loss, step_time):
        print(f'step: {step}, loss: {loss}, step_step_time: {step_time}')

    def step_fn(self, pixel_values, input_ids):
        self.optimizer.zero_grad()
        with xp.Trace("model.forward"):
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            noise = torch.randn_like(latents).to(self.device, dtype=self.weight_dtype)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states = self.text_encoder(input_ids, return_dict=False)[0]
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
            model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        with xp.Trace("model.backward"):
            loss.backward()
        self.run_optimizer()

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
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
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    args = parser.parse_args()
    return args

def setup_model_parallel(ddp=False) -> Tuple[int, int]:
    rank = xm.get_ordinal()
    world_size = xm.xrt_world_size()
    if ddp:
        torch.distributed.init_process_group("xla", init_method="xla://")

    # seed must be the same in all processes
    torch.manual_seed(1)
    device = xm.xla_device()
    xm.set_rng_state(1, device=device)
    return rank, world_size

def setup_optimizer(unet, args):
    optimizer_cls = torch.optim.AdamW
    return optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

def main(args):

    args = parse_args()

    server = xp.start_server(9012)

    num_devices = xr.global_runtime_device_count()
    device_ids = np.arange(num_devices)
    mesh_shape = (num_devices, 1, 1)
    mesh = xs.Mesh(device_ids, mesh_shape, ('batch', 'fsdp', 'tensor'))
    xs.set_global_mesh(mesh)

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder",
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet",
    )
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer",
    )
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()
    optimizer = setup_optimizer(unet, args)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    device = xm.xla_device()
    print("device: ", device)
    print("weight_dtype: ", weight_dtype)

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder = text_encoder.to(device, dtype=weight_dtype)
    vae = vae.to(device, dtype=weight_dtype)
    # vae_compiled = torch.compile(vae, backend="openxla")
    unet = unet.to(device, dtype=weight_dtype)
    # unet_compiled = torch.compile(unet, backend="openxla")

    if xm.is_master_ordinal():
        print("***** Running training *****")
        print(f"  Num Epochs = {args.num_train_epochs}")
        print(f"  Instantaneous batch size per device = {args.train_batch_size // num_devices}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {args.train_batch_size}")
        print(f"  Total optimization steps = {args.num_train_epochs*args.max_train_steps}")

    trainer = TrainSD(vae=vae,
                      weight_dtype=weight_dtype,
                      device=device,
                      noise_scheduler=noise_scheduler,
                      unet=unet,
                      optimizer=optimizer,
                      text_encoder=text_encoder,
                      args=args)
    
    trainer.start_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)
    #xmp.spawn(main,args=())
