# 排队脚本（仅作记录）

这些是实验实际执行时用的排队脚本，路径（`~/Programs/dinov3-crop-vit`、`~/data_crops`、`/data/...`、`D:\dinov3-crop-vit`）和 conda 环境名都写死了，**不能直接运行**，保留下来是为了说明每个 run 的准确参数和先后顺序。

| 脚本 | 内容 |
|---|---|
| `lab/queue_crops.sh` | 5 个数据集第一版（紧框）裁剪 |
| `lab/queue_train.sh` | 紧框版本的 14 个 run（跑到 car 时发现掉分而停止） |
| `lab/queue_train_sq.sh` | 改用正方形 + 15% 上下文（`_sq`）后的全部 run |
| `lab/queue_224_probe.sh`, `lab/queue_224b.sh` | CUB 224 分辨率探针 |
| `lab/queue_s1.sh`, `lab/queue_nab_s1.sh` | seed 1 复现 |
| `lab/queue_tok.sh` | token 选择：先跑 smoke test，达标后自动排完整实验（决策记录在 `results/lab/logs/tok_decision.log`） |
| （Windows 3060 上的 bat 脚本未收录，见 results/RESULTS.md 末尾的 5 个 run） |
