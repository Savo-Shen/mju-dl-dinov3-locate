#!/bin/bash
# token 选择变体：先在空闲卡上跑 smoke test（CUB，DINOv3 选 25% token，完整 10k 步），
# 结果 ≥ 88.5（raw 90.68 的 -2.2 以内，token 只有 1/4）就自动排完整实验；否则记录后停止。
source /opt/anaconda3/etc/profile.d/conda.sh; conda activate fal-wyh
cd ~/Programs/dinov3-crop-vit; mkdir -p logs output
export HF_ENDPOINT=https://hf-mirror.com
D=~/data_crops; LOG=logs/tok_decision.log
foreign_on() {  # GPU $1 上有不属于本目录的计算进程？
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $1); do
    tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q "dinov3-crop-vit\|train_vit" || return 0
  done; return 1
}
pick_gpu() {  # 优先 GPU0（学长的卡，现在空着），被占就等 GPU1 的队列结束
  while true; do
    # GPU0 上常驻一个 gnome-remote-desktop 进程（~260MB），只看显存占用判断是否空闲
    if [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)" -lt 3000 ]; then echo 0; return; fi
    if [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)" -lt 3000 ] && ! pgrep -f "queue_s[1]" >/dev/null; then echo 1; return; fi
    sleep 120
  done
}
run() {  # run NAME DATA_SUBDIR EXTRA...
  local N=$1 P=$2; shift 2
  [ -f output/$N.json ] && { echo "skip $N (done)" | tee -a $LOG; return; }
  local G=$(pick_gpu)
  echo "=== $N on GPU$G start $(date) ===" | tee -a $LOG
  CUDA_VISIBLE_DEVICES=$G python train_vit_tok.py --data_root $D/$P --name $N "$@" > logs/train_$N.log 2>&1
  grep -a "Best Accuracy" logs/train_$N.log | tail -1 | tee -a $LOG
}
best() { python -c "import json,sys; print(json.load(open('output/$1.json'))['best'])" 2>/dev/null || echo 0; }

run cub_tok25 cub/raw --select dinov3 --keep 0.25
B=$(best cub_tok25)
echo "smoke cub_tok25 best=$B (raw 0.9068; threshold 0.885)" | tee -a $LOG
if python -c "import sys; sys.exit(0 if $B >= 0.885 else 1)"; then
  echo "DECISION: effective → full token-selection set" | tee -a $LOG
  run cub_tok50        cub/raw     --select dinov3 --keep 0.50
  run cub_tok10        cub/raw     --select dinov3 --keep 0.10
  run cub_rand25       cub/raw     --select random --keep 0.25
  run cub_tok25_896    cub/raw     --select dinov3 --keep 0.25 --img_size 896 --train_batch_size 8 --grad_accum 2 --eval_batch_size 32
  run nabirds_tok25    nabirds/raw --select dinov3 --keep 0.25
  run nabirds_rand25   nabirds/raw --select random --keep 0.25
  run nabirds_tok25_896 nabirds/raw --select dinov3 --keep 0.25 --img_size 896 --train_batch_size 8 --grad_accum 2 --eval_batch_size 32
  run nabirds_tok10    nabirds/raw --select dinov3 --keep 0.10
else
  echo "DECISION: not effective (best=$B < 0.885) → stop" | tee -a $LOG
fi
echo "=== tok queue ALL DONE $(date) ===" | tee -a $LOG
