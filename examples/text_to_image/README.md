```
git clone -b ptxla_sd2 https://github.com/entrpn/diffusers
cd diffusers
cd examples/text_to_image
pip install -r requirements.txt
cd ../..
pip install -e .
export MODEL_NAME=stabilityai/stable-diffusion-2-base
export DATASET_NAME="lambdalabs/naruto-blip-captions"
PROFILE_DIR=gs://bbahl/sd2 CACHE_DIR=/home/bbahl/cache XLA_HLO_DEBUG=1 XLA_IR_DEBUG=1 python train_text_to_image.py  --pretrained_model_name_or_path=$MODEL_NAME --dataset_name=$DATASET_NAME --resolution=512 --center_crop --random_flip   --train_batch_size=64  --max_train_steps=20   --learning_rate=1e-06  --mixed_precision="bf16"   --output_dir="sdxl-naruto-model"
```