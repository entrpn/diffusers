from time import perf_counter
from pathlib import Path
from argparse import ArgumentParser

import structlog
import time

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.debug.profiler as xp
import torch_xla.debug.metrics as met

from diffusers import FluxPipeline

logger = structlog.get_logger()
metrics_filepath = '/tmp/metrics_report.txt'

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--schnell', action='store_true', help='run flux schnell instead of dev')
    parser.add_argument('--width', type=int, default=1024, help='width of the image to generate')
    parser.add_argument('--height', type=int, default=1024, help='height of the image to generate')
    parser.add_argument('--guidance', type=float, default=3.5, help='gauidance strentgh for dev')
    parser.add_argument('--seed', type=int, default=None, help='seed for inference')
    parser.add_argument('--profile', action='store_true', help='enable profiling')
    parser.add_argument('--profile-duration', type=int, default=10000, help='duration for profiling in msec.')
    args = parser.parse_args()

    cache_path = Path('/tmp/data/compiler_cache')
    cache_path.mkdir(parents=True, exist_ok=True)
    xr.initialize_cache(str(cache_path), readonly=False)

    profile_path = Path('/tmp/data/profiler_out')
    profile_path.mkdir(parents=True, exist_ok=True)
    profiler_port = 9012
    profile_duration = args.profile_duration
    if args.profile:
        logger.info(f'starting profiler on port {profiler_port}')
        _ = xp.start_server(profiler_port)

    device0 = xm.xla_device(0)
    device1 = xm.xla_device(1)
    logger.info(f'text encoders: {device0}, flux: {device1}')

    if args.schnell:
        ckpt_id = "black-forest-labs/FLUX.1-schnell"
    else:
        ckpt_id = "black-forest-labs/FLUX.1-dev"
    logger.info(f'loading flux from {ckpt_id}')

    text_pipe = FluxPipeline.from_pretrained(ckpt_id, transformer=None, vae=None, torch_dtype=torch.bfloat16).to(device0)
    flux_pipe = FluxPipeline.from_pretrained(ckpt_id, text_encoder=None, tokenizer=None,
                                            text_encoder_2=None, tokenizer_2=None, torch_dtype=torch.bfloat16).to(device1)

    prompt = 'photograph of an electronics chip in the shape of a race car with trillium written on its side'
    width = args.width
    height = args.height
    guidance = args.guidance
    n_steps = 4 if args.schnell else 28

    logger.info('starting compilation run...')
    ts = perf_counter()
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = text_pipe.encode_prompt(
            prompt=prompt, prompt_2=None, max_sequence_length=512)
    prompt_embeds = prompt_embeds.to(device1)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device1)

    image = flux_pipe(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
                    num_inference_steps=28, guidance_scale=guidance, height=height, width=width).images[0]
    logger.info(f'compilation took {perf_counter() - ts} sec.')
    image.save('/tmp/compile_out.png')

    seed = 0 if args.seed is None else args.seed
    xm.set_rng_state(seed=seed, device=device0)
    xm.set_rng_state(seed=seed, device=device1)

    logger.info('starting inference run...')
    ts = perf_counter()
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = text_pipe.encode_prompt(
            prompt=prompt, prompt_2=None, max_sequence_length=512)
    prompt_embeds = prompt_embeds.to(device1)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device1)
    xm.wait_device_ops()

    if args.profile:
        xp.trace_detached(f"localhost:{profiler_port}", str(profile_path), duration_ms=profile_duration)
    image = flux_pipe(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
                    num_inference_steps=n_steps, guidance_scale=guidance, height=height, width=width).images[0]
    logger.info(f'inference took {perf_counter() - ts} sec.')
    image.save('/tmp/inference_out.png')
    metrics_report = met.metrics_report()
    with open(metrics_filepath, 'w+') as fout:
        fout.write(metrics_report)
    logger.info(f'saved metric information as {metrics_filepath}')

