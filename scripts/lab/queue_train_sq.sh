#!/bin/bash
# 队列 2（09-16 02:20 起）：紧框裁剪已在 CUB/Dogs 上证明会掉分（细长框被 Resize((600,600)) 拉变形 + 丢上下文），
# 而正方形+15% 上下文的 _sq 变体在 CUB 上追平 raw（90.58 vs 90.68）。剩余数据集全部改跑 _sq：
#   1) 等旧队列当前的 run 结束后接管 GPU1（旧 queue_train.sh 已被 kill，不会再起新 run）
#   2) 用 --square --margin 0.15 --suffix _sq 重做各数据集裁剪（cub 的 _sq 在 3060 上跑，这里不重复）
#   3) 训 dinov3_sq / gt_sq；nabirds、ip102 的 raw 还没跑，也在这里跑
source /opt/anaconda3/etc/profile.d/conda.sh; conda activate fal-wyh
cd ~/Programs/dinov3-crop-vit; mkdir -p logs output
export HF_ENDPOINT=https://hf-mirror.com
D=~/data_crops
busy() { pgrep -f "train_vit.py" > /dev/null; }
while busy; do sleep 60; done
echo "=== queue2 takes GPU1 $(date) ==="
for DS in car dog nabirds ip102; do
  [ -f $D/$DS/boxes_sq.csv ] && continue
  echo "=== crops_sq $DS start $(date) ==="
  CUDA_VISIBLE_DEVICES=1 python make_crops.py --dataset $DS --tau 0.07 --margin 0.15 --square --suffix _sq > logs/crops_${DS}_sq.log 2>&1 &
  CROP_PID=$!
  # 裁剪与训练共卡：裁剪只占 ~4GB，先让第一个可跑的训练起来
  break
done
run() {  # run NAME DATA_SUBDIR
  [ -f output/$1.json ] && { echo "skip $1 (done)"; return; }
  while [ ! -d $D/$2/train ]; do sleep 60; done
  echo "=== train $1 start $(date) ==="
  CUDA_VISIBLE_DEVICES=1 python train_vit.py --data_root $D/$2 --name $1 > logs/train_$1.log 2>&1
  grep -a "Best Accuracy" logs/train_$1.log | tail -1
}
crops_sq() {  # 顺序做完剩余数据集的 _sq 裁剪（后台，与训练共卡）
  for DS in car dog nabirds ip102; do
    [ -f $D/$DS/boxes_sq.csv ] && continue
    echo "=== crops_sq $DS start $(date) ==="
    CUDA_VISIBLE_DEVICES=1 python make_crops.py --dataset $DS --tau 0.07 --margin 0.15 --square --suffix _sq > logs/crops_${DS}_sq.log 2>&1
    grep -a "IoU\|done" logs/crops_${DS}_sq.log
  done
}
wait $CROP_PID 2>/dev/null
crops_sq &
run car_dinov3_sq   car/dinov3_sq
run car_gt_sq       car/gt_sq
run dog_dinov3_sq   dog/dinov3_sq
run dog_gt_sq       dog/gt_sq
run nabirds_raw     nabirds/raw
run nabirds_dinov3_sq nabirds/dinov3_sq
run nabirds_gt_sq   nabirds/gt_sq
run ip102_raw       ip102/raw
run ip102_dinov3_sq ip102/dinov3_sq
wait
echo "=== queue2 ALL DONE $(date) ==="
