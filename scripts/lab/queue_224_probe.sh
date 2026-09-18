#!/bin/bash
# 224 分辨率探针：ViT 看到的像素少 4 倍时，先找主体再放大是否变得有用（CUB raw vs dinov3_sq vs gt_sq）
source /opt/anaconda3/etc/profile.d/conda.sh; conda activate fal-wyh
cd ~/Programs/dinov3-crop-vit
for V in raw dinov3_sq gt_sq; do
  N=cub_${V}_224; [ -f output/$N.json ] && continue
  echo "=== train $N start $(date) ==="
  CUDA_VISIBLE_DEVICES=1 python train_vit.py --data_root ~/data_crops/cub/$V --name $N --img_size 224 --train_batch_size 16 --eval_batch_size 128 > logs/train_$N.log 2>&1
  grep -a "Best Accuracy" logs/train_$N.log | tail -1
done
echo "=== 224 probe ALL DONE $(date) ==="
