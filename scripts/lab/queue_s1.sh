#!/bin/bash
# 补 seed 1：dog/car 的 raw（顺便拿到逐图预测做按主体大小分组）→ NABirds raw/dinov3_sq（确认 +1.4）
source /opt/anaconda3/etc/profile.d/conda.sh; conda activate fal-wyh
cd ~/Programs/dinov3-crop-vit
for J in "dog_raw_s1|dog/raw" "car_raw_s1|car/raw" "nabirds_raw_s1|nabirds/raw" "nabirds_dinov3_sq_s1|nabirds/dinov3_sq" "dog_dinov3_sq_s1|dog/dinov3_sq" "car_dinov3_sq_s1|car/dinov3_sq"; do
  IFS="|" read -r N P <<< "$J"; [ -f output/$N.json ] && continue
  echo "=== train $N start $(date) ==="
  CUDA_VISIBLE_DEVICES=1 python train_vit.py --data_root ~/data_crops/$P --name $N --seed 1 > logs/train_$N.log 2>&1
  grep -a "Best Accuracy" logs/train_$N.log | tail -1
done
echo "=== s1 ALL DONE $(date) ==="
