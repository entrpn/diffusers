# Stable Diffusion XL text-to-image fine-tuning using PyTorch/XLA

The `train_text_to_image_sdxl.py` script shows how to fine-tune stable diffusion model on TPU devices using PyTorch/XLA.

It has been tested on v5p TPU versions. Training code has been tested on a single v5p-8.

This script implements Distributed Data Parallel using GSPMD feature in XLA compiler
where we shard the input batches over the TPU devices. 

As of 04-04-2025, these are some expected step times.

| accelerator | global batch size | step time (seconds) |
| ----------- | ----------------- | --------- |
| v5p-8 | 32 | 1.02 |
| v5p-8 | 48 (with gradient checkpointing) | 1.42 |
| v5p-8 | 64 (with gradient checkpointing) | 1.66 |
|
## Create TPU

To create a TPU on Google Cloud first set these environment variables:

```bash
export TPU_NAME=<tpu-name>
export PROJECT_ID=<project-id>
export ZONE=<google-cloud-zone>
export ACCELERATOR_TYPE=<accelerator type like v5p-8>
export RUNTIME_VERSION=<runtime version like v2-alpha-tpuv5 for v5p>
```

Then run the create TPU command:
```bash
gcloud alpha compute tpus tpu-vm create ${TPU_NAME} --project ${PROJECT_ID} 
--zone ${ZONE} --accelerator-type ${ACCELERATOR_TYPE} --version ${RUNTIME_VERSION} 
--reserved
```

You can also use other ways to reserve TPUs like GKE or queued resources.

## Setup TPU environment

Assuming that you have conda setup on the VM
Install PyTorch and PyTorch/XLA nightly versions:
```bash
conda create -n torch310 python=3.10
conda activate torch310
pip install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch-2.8.0a0%2Bgit01cb351-cp310-cp310-linux_x86_64.whl
pip install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.8.0%2Bgite341ff0-cp310-cp310-linux_x86_64.whl
pip install https://storage.googleapis.com/libtpu-nightly-releases/wheels/libtpu/libtpu-0.0.11.dev20250303+nightly-py3-none-manylinux_2_27_x86_64.whl
pip install torchax
pip install jax==0.5.4.dev20250321 jaxlib==0.5.4.dev20250321 \
    -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
pip install --no-deps torchvision --index-url https://download.pytorch.org/whl/nightly/cpu

git clone -b sdxl_training_bbahl git@github.com:entrpn/diffusers.git
cd diffusers/
pip install -r examples/research_projects/pytorch_xla/training/text_to_image/requirements_sdxl.txt

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ${CMAKE_PREFIX_PATH}/lib/libstdc++.so.6

pip install -e .
```

## Run the training job

Run the following command to authenticate your token.

```bash
huggingface-cli login
```

Please update the variables in `train.sh` as needed

```bash
export XLA_DISABLE_FUNCTIONALIZATION=1
export TORCH_DISABLE_FUNCTIONALIZATION_META_REFERENCE=1
export PROFILE_DIR=/tmp/xla_profile/
export CACHE_DIR=/tmp/xla_cache/
export DATASET_NAME=lambdalabs/naruto-blip-captions
export PER_HOST_BATCH_SIZE=64 # This is known to work on TPU v5p with gradient checkpointing.
export TRAIN_STEPS=50
export PROFILE_START_STEP=10
export OUTPUT_DIR=/tmp/docker/trained-model/
export HF_HOME="/tmp/hf_home/"
# export XLA_HLO_DEBUG=1
# export XLA_IR_DEBUG=1
python examples/research_projects/pytorch_xla/training/text_to_image/train_text_to_image_sdxl.py --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 --dataset_name=$DATASET_NAME --resolution=1024 --center_crop --random_flip --train_batch_size=$PER_HOST_BATCH_SIZE  --max_train_steps=$TRAIN_STEPS --measure_start_step=$PROFILE_START_STEP --learning_rate=1e-06 --mixed_precision=bf16 --profile_duration=5000 --output_dir=$OUTPUT_DIR --dataloader_num_workers=8 --loader_prefetch_size=4 --device_prefetch_size=4 --xla_gradient_checkpointing
```


This script only trains the unet part of the network. The VAE and text encoder
are fixed.

```bash
./train.sh
```

Pass `--print_loss` if you would like to see the loss printed at every step. Be aware that printing the loss at every step disrupts the optimized flow execution, thus the step time will be longer. 

### Environment Envs Explained

*   `XLA_DISABLE_FUNCTIONALIZATION`: To optimize the performance for AdamW optimizer.
*   `PROFILE_DIR`: Specify where to put the profiling results.
*   `CACHE_DIR`: Directory to store XLA compiled graphs for persistent caching.
*   `DATASET_NAME`: Dataset to train the model. 
*   `PER_HOST_BATCH_SIZE`: Size of the batch to load per CPU host. For e.g. for a v5p-16 with 2 CPU hosts, the global batch size will be 2xPER_HOST_BATCH_SIZE. The input batch is sharded along the batch axis.
*    `TRAIN_STEPS`: Total number of training steps to run the training for.
*    `OUTPUT_DIR`: Directory to store the fine-tuned model.

## Run inference using the output model

To run inference using the output, you can simply load the model and pass it
input prompts. The first pass will compile the graph and takes longer with the following passes running much faster.

```bash
export CACHE_DIR=/tmp/
export OUTPUT_DIR=/tmp/trained-model
python inference_sdxl.py
```
Expected Results:

```bash
compiling...
  0%|                                                                                                                                                                                                                                      | 0/30 [00:00<?, ?it/s]/mnt/bbahl/miniconda3/envs/torchverify/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:351: UserWarning: Device capability of jax unspecified, assuming `cpu` and `cuda`. Please specify it via the `devices` argument of `register_backend`.
  warnings.warn(
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [01:29<00:00,  2.97s/it]
compile time: 226.64892053604126
generate...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:04<00:00,  6.17it/s]
generation time (after compile) : 5.120622396469116
```