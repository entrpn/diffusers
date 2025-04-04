import torch
import os
import sys
import  numpy as np

import torch_xla.core.xla_model as xm
from time import time
from diffusers import StableDiffusionXLPipeline
import torch_xla.runtime as xr

CACHE_DIR = os.environ.get("CACHE_DIR", '/mnt/bbahl/xla_cache/')
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", '/mnt/bbahl/trained-model/')
if CACHE_DIR:
    xr.initialize_cache(CACHE_DIR, readonly=False)


device = xm.xla_device()
model_path = OUTPUT_DIR
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16
)
pipe.to(device)
prompt = ["A naruto with green eyes and red legs."]

pipe.unet.enable_xla_attention()
# pipe.vae.enable_xla_attention()
start = time()
print("compiling...")
import pdb; pdb.set_trace()
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
print(f"compile time: {time() - start}")
print("generate...")
start = time()
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
print(f"generation time (after compile) : {time() - start}")
image.save("naruto.png")