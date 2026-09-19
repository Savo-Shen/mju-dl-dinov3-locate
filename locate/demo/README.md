# locate/demo/ — 现场演示（合并版）

```bash
conda activate dinov3 && export HF_ENDPOINT=https://hf-mirror.com
python locate/demo/server.py        # → http://127.0.0.1:8001
```

一个页面、两个页签，同一张图点一次"分析"两边都算：

- **① 找主体 → 分类**：DINOv3 注意力找框 → 三个 NABirds 分类器并排预测（本目录）
- **② 三个冻结 ViT 在看什么**：dinov2 / dinov2_reg4 / dinov3 的 patch 特征 PCA 着色图 + NCut 前景 mask（原 `experiments/app`，逻辑在 `experiments/app/features.py`，也可单独用 `experiments/app/run.sh` 起）

页面右上角两个说明按钮：**"DINOv3 的注意力图是怎么来的？"**（CLS token → 最后一层 CLS 对各 patch 的注意力 → 12 头平均 → 排回 28×28；为什么亮处是主体；register 为什么让它干净；和 NCut mask 的区别）和 **"页签② 的图怎么看？"**。

页签① 上传（或拖入、粘贴）一张鸟的照片后，一行四张图、下面三列预测：

| 图 | 内容 |
|---|---|
| ① 原图 + 自动框 | DINOv3 不用标注框出的主体（红框），标注主体占图比例 |
| DINOv3 CLS 注意力 | 找框依据：越亮越"像主体" |
| ② 裁剪放大 | 正方形框 + 15% 上下文，送给"裁剪分类器" |
| ③ ViT 实际看到的 token | 只保留 DINOv3 挑出的 196 / 784 个 patch，其余变暗 |

| 列 | 模型 | 对应汇报里的 |
|---|---|---|
| 原图 | `weights/nabirds_raw_ckpt.pt`，看整张图 | 基线 |
| DINOv3 裁剪 | `weights/nabirds_dinov3_sq_ckpt.pt`，看②  | 方向一：主体看不清 → 先找再放大 |
| 只看 25% token | `weights/nabirds_tok25_ckpt.pt`，只看③ | 方向二：主体看得清 → 换算力 |

三个都是 NABirds（555 类）上微调的 ViT-B/16，训练命令见根目录 README §5–§6（加 `--save_ckpt`）；缺哪个权重那一列就显示"未就绪"，其余照常。找框和挑 token 的参数（τ=0.07、正方形 + 15%、keep 25%、3×3 平滑）与训练时完全一致，所以现场结果和 `results/` 里的逐图预测可以对上。

M1 Max（MPS）实测：找框约 50 ms，每个分类器约 50 ms，页签① 一张图 0.2 秒以内；页签② 三个 ViT-S 约 0.6 秒。全部模型（DINOv3-B、三个 ViT-B 分类器、三个 ViT-S）加载约 15 秒、占内存约 1.5 GB。

示例图放在 `data/demo_samples/` 和 `experiments/data/`（都不入库；我们用的是 NABirds 测试集里"原图错、裁剪对"的几张，按 `编号_鸟名_主体占比.jpg` 命名，文件名会显示成按钮）。
