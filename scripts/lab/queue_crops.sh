#!/bin/bash
# 阶段 1 队列：5 个数据集依次用 DINOv3 找框裁剪（GPU1，与训练队列共卡，显存占用小）。
# 校准结论（CUB 训练集 1000 张 vs 官方 bbox）：attn τ=0.07 margin=0 → mean IoU 0.792，IoU≥0.5 占 96.8%。
source /opt/anaconda3/etc/profile.d/conda.sh; conda activate fal-wyh
cd ~/Programs/dinov3-crop-vit; mkdir -p logs
export HF_ENDPOINT=https://hf-mirror.com
for DS in cub dog car nabirds ip102; do
  echo "=== crops $DS start $(date) ==="
  CUDA_VISIBLE_DEVICES=1 python make_crops.py --dataset $DS --tau 0.07 --margin 0 > logs/crops_$DS.log 2>&1
  grep -a "IoU\|done" logs/crops_$DS.log
done
echo "=== crops ALL DONE $(date) ==="
