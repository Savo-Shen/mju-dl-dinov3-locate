#!/usr/bin/env bash
# 一条命令起 Demo：bash experiments/app/run.sh   （先 conda activate dinov3）
cd "$(dirname "$0")/../.."
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
exec python experiments/app/server.py
