# experiments/ — Mac 本机的 Demo 与前置实验

在 M1 Max 上用 MPS 跑的部分，只用 21M 的 ViT-S 权重。目录布局固定（脚本用 `experiments/` 相对路径找数据和写产物），把 CUB 放到 `experiments/data/CUB/` 后直接运行。

| 子目录 | 内容 |
|---|---|
| `app/` | Web Demo：`bash experiments/app/run.sh` → http://127.0.0.1:8000。三模型（dinov2 / dinov2_reg4 / dinov3）并排的 PCA 着色图与前景 mask，可上传图片。核心逻辑在 `app/features.py`，`locate/demo/` 的合并版 Demo 把它作为页签② 复用 |
| `scripts/` | 实验 01–03 的脚本，见 `scripts/README.md` |
| `notes/` | 每个实验一份笔记：做法、数字、结论，以及方法上踩过的坑 |
| `outputs/` | 实验产物：表（csv/md）、图、日志；300MB 的特征缓存（`fgvc_bg/feats/`）没有入库，重跑 `fgvc_bg.py extract` 可再生成 |
| `data/` | 放 `CUB/CUB_200_2011/` 和 `CUB/segmentations/`（不入库） |

三个前置实验和它们对主实验的影响：

1. **01 PCA 可视化**——DINOv3 的 patch 特征把主体和背景分得比 DINOv2 干净 → 值得拿它来定位。
2. **02 1000 张前景 IoU**——DINOv3 0.681 vs DINOv2 0.560，9 个分组全部领先 → 定位能力是普遍的，不是个例。
3. **03 冻结特征线性探针**——三代分类都 84%，涂掉背景不涨，但换成 mean-pool 时涂背景反涨 12 点 → ViT 的 CLS 注意力自己就在定位，所以主实验里的裁剪必须是"放大"而不是"去背景"。
