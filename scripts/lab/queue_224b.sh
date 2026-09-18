#!/bin/bash
# 224 探针补跑：先在 lab 上生成 CUB 的 _sq（448 找框）和 _sq224（DINOv3 只看 224 找框，省 3/4 算力）裁剪，再训 224 的三个变体
source /opt/anaconda3/etc/profile.d/conda.sh; conda activate fal-wyh
cd ~/Programs/dinov3-crop-vit; export HF_ENDPOINT=https://hf-mirror.com
[ -f ~/data_crops/cub/boxes_sq.csv ] || CUDA_VISIBLE_DEVICES=1 python make_crops.py --dataset cub --tau 0.07 --margin 0.15 --square --suffix _sq > logs/crops_cub_sq.log 2>&1
[ -f ~/data_crops/cub/boxes_sq224.csv ] || CUDA_VISIBLE_DEVICES=1 python make_crops.py --dataset cub --tau 0.07 --margin 0.15 --square --suffix _sq224 --img 224 > logs/crops_cub_sq224.log 2>&1
grep -a "IoU(auto, gt) all" logs/crops_cub_sq.log logs/crops_cub_sq224.log
for V in dinov3_sq gt_sq dinov3_sq224; do
  N=cub_${V}_224; [ -f output/$N.json ] && continue
  echo "=== train $N start $(date) ==="
  CUDA_VISIBLE_DEVICES=1 python train_vit.py --data_root ~/data_crops/cub/$V --name $N --img_size 224 --train_batch_size 16 --eval_batch_size 128 > logs/train_$N.log 2>&1
  grep -a "Best Accuracy" logs/train_$N.log | tail -1
done
echo "=== 224b ALL DONE $(date) ==="
