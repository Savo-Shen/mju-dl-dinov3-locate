#!/bin/bash
# 阶段 2 队列：纯 ViT-B/16 在 raw / dinov3 / gt 三种数据上微调，每个数据集等其 boxes.csv 出现后再开始。
source /opt/anaconda3/etc/profile.d/conda.sh; conda activate fal-wyh
cd ~/Programs/dinov3-crop-vit; mkdir -p logs output
D=~/data_crops
for DS in cub dog car nabirds ip102; do
  while [ ! -f $D/$DS/boxes.csv ]; do sleep 60; done
  for V in raw dinov3 gt; do
    [ -d $D/$DS/$V/train ] || continue
    [ -f output/${DS}_${V}.json ] && { echo "skip ${DS}_${V} (done)"; continue; }
    echo "=== train ${DS}_${V} start $(date) ==="
    CUDA_VISIBLE_DEVICES=1 python train_vit.py --data_root $D/$DS/$V --name ${DS}_${V} > logs/train_${DS}_${V}.log 2>&1
    grep -a "Best Accuracy" logs/train_${DS}_${V}.log | tail -1
  done
done
echo "=== train ALL DONE $(date) ==="
