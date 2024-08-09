from diffusers import StableDiffusionPipeline
import torch
import os
import sys
import torch
import  numpy as np
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import torch_xla.distributed.xla_multiprocessing as xmp
from time import time
from typing import Tuple
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers import DiffusionPipeline

def setup_model_parallel() -> Tuple[int, int]:
    # assuming model parallelism over the whole world size
    rank = xm.get_ordinal()
    world_size = xm.xrt_world_size()

    # seed must be the same in all processes
    torch.manual_seed(1)
    device = xm.xla_device()
    xm.set_rng_state(1, device=device)
    return rank, world_size

def main(args):

    server = xp.start_server(9012)
    device = xm.xla_device()
    rank, world_size = setup_model_parallel()
    print('rank, world_size', rank, world_size)

    # print only for xla:0 device
    if rank > 0:
        sys.stdout = open(os.devnull, "w")

    # not really sdxl model, its sd 2 base.
    model_path = "sdxl-naruto-model"
    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.to(device)

    prompt = ["A naruto with green eyes and red legs."] * world_size
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    image.save("naruto.png")


if __name__ == '__main__':
    xmp.spawn(main, args=())