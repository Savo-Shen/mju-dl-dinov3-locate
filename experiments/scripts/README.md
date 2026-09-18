# experiments/scripts/

| 脚本 | 实验 | 用法 | 产物 |
|---|---|---|---|
| `vitfeat.py` | 公共 | 三个模型的加载与 patch 特征提取（timm，MPS/CPU），Demo 和下面的脚本都 import 它 | — |
| `pca_features.py` | 01 | `python experiments/scripts/pca_features.py --images experiments/data --mask` | `outputs/pca_v2_vs_v3.jpg` |
| `bg_quant.py` | 02 | `python experiments/scripts/bg_quant.py --n 1000`（`--analyze-only` 只重跑统计） | `outputs/bg_quant/`：逐图 csv、报告 md、图 |
| `fgvc_bg.py` | 03 | `… fgvc_bg.py extract`（抽特征，约 20 分钟）→ `… fgvc_bg.py probe`（线性探针 + kNN） | `outputs/fgvc_bg/`：准确率表、报告、图 |
| `fgvc_evidence.py` | 03 补充 | `python experiments/scripts/fgvc_evidence.py --n 500 --viz 8`：分类证据热力图 | `outputs/fgvc_bg/evidence/` |

方法上的坑（详见 `notes/`）：单图 PC1 不是前景轴，前景要用 NCut；TokenCut 的固定阈值对 DINOv3 失效，改为每图中位数；ViT-S 没有高范数 token，artifact 指标作废。
