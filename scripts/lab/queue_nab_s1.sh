#!/bin/bash
# NABirds 是唯一有明显涨幅（+1.4）的数据集，补 seed 1 确认
source /opt/anaconda3/etc/profile.d/conda.sh; conda activate fal-wyh
cd ~/Programs/dinov3-crop-vit
for V in raw dinov3_sq gt_sq; do
  N=nabirds_${V}_s1; [ -f output/$N.json ] && continue
  echo "=== train $N start $(date) ==="
  CUDA_VISIBLE_DEVICES=1 python train_vit.py --data_root ~/data_crops/nabirds/$V --name $N --seed 1 > logs/train_$N.log 2>&1
  grep -a "Best Accuracy" logs/train_$N.log | tail -1
done
echo "=== nab_s1 ALL DONE $(date) ==="
