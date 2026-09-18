#!/bin/bash
source /opt/anaconda3/etc/profile.d/conda.sh; conda activate fal-wyh
cd ~/Programs/dinov3-crop-vit; export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=0 python train_vit_tok.py --data_root ~/data_crops/cub/raw --name _smoke_tmp --select dinov3 --keep 0.25 --num_steps 30 --eval_every 1000 --warmup_steps 5 --output_dir /tmp/tok_tmp > logs/_smoke_check.log 2>&1
echo "exit=$?" >> logs/_smoke_check.log
