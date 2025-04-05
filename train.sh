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
