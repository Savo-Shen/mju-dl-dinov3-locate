# results/

服务器上全部 35 个 run 的原始结果，足以不重跑就复算 README 里的每个数字。

| 文件 | 内容 |
|---|---|
| `RESULTS.md` | 由 json 自动生成的总表（best / final / 设置），以及 RTX 3060 上跑的 5 个 CUB run |
| `lab/output/<name>.json` | 每个 run：best、final、每 1000 步的准确率曲线、全部命令行参数 |
| `lab/output/<name>_preds.npz` | 最佳一轮的逐图预测与标签（int16）；为省空间去掉了路径，路径在同目录 `test_paths_<ds>.txt`（同一数据集所有 run 顺序相同，= ImageFolder 排序） |
| `lab/boxes/<ds>_test_boxes_sq.csv.gz` | 测试集每张图的 DINOv3 自动框、官方 bbox、IoU、原图尺寸（`make_crops.py` 产出的 `boxes_sq.csv` 的 test 部分） |
| `lab/logs/train_<name>.log` | 训练日志（每 100 步 loss、每 1000 步准确率） |
| `lab/logs/crops_<ds>[_sq].log` | 裁剪日志，末尾有自动框 vs 官方框的 IoU 统计 |
| `lab/tok_decision.log` | token 选择实验的自动决策记录（smoke test → 达标 → 排完整实验） |

命名：`<数据集>_<输入变体>[_设置][_s1]`。输入变体 `raw` 原图 / `dinov3` `gt` 第一版紧框（已弃用）/ `dinov3_sq` `gt_sq` 正方形 + 15% 上下文（最终版）；设置 `_224` 输入 224、`tok25` DINOv3 挑 25% token、`rand25` 随机挑 25%、`tok25_896` 896 输入挑 25%；`_s1` = seed 1（默认 seed 42）。

复算按主体大小分组的表：

```bash
python locate/analyze_size.py nabirds results/lab/boxes/nabirds_test_boxes_sq.csv.gz results/lab/output nabirds_raw nabirds_dinov3_sq nabirds_gt_sq
```
