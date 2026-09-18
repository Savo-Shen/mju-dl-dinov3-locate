# locate/ — 主实验代码

"DINOv3 找主体 → 纯 ViT 分类"的全部代码，六个脚本、一条流水线。都从仓库根目录运行，数据在 `data/`（或 `DATA_ROOT`），权重在 `weights/`。

| 脚本 | 阶段 | 做什么 | 输入 → 输出 |
|---|---|---|---|
| `dino_attn.py` | 公共 | 从冻结的 DINOv3（timm）里读最后一层 CLS→patch 注意力，作为"主体在哪"的分数；另两个脚本 import 它 | 图 (B,3,448,448) → 注意力图 (B,28,28) + patch 特征 |
| `make_crops.py` | 1 | 注意力图 → 阈值 τ → 最大连通块外接框 → 补正方形、外扩 15% → 从原图裁剪；同时按官方 bbox 裁一份当上界，算 IoU | 原始数据集 → `data/crops/<ds>/{raw,dinov3_sq,gt_sq}/{train,test}/<cls>/` + `boxes_sq.csv` |
| `train_vit.py` | 2 | 纯 ViT-B/16（ImageNet-21k）微调，TransFG 配方（600→448 crop，SGD 3e-2，10k 步）；`--img_size 224` 做分辨率探针 | ImageFolder → `output/<name>.json`（best、每千步曲线）+ `output/<name>_preds.npz`（逐图预测） |
| `train_vit_tok.py` | 2′ | 同上，但 patch embedding 后只保留 DINOv3 分数最高的 `--keep` 比例的 token（`--select random` 做对照，`--img_size 896` 先放大再挑） | 同上，json 里多一项 `tokens: [k, N]` |
| `analyze_size.py` | 分析 | 按官方 bbox 面积占比把测试集等频分 5 组，比较各 run 的准确率 | `boxes.csv(.gz)` + 若干 `_preds.npz` → 终端表格 |
| `viz_boxes.py` | 分析 | 随机抽 8 张 + IoU 最低 2 张，画红框（自动）/ 青框（官方） | → `/tmp/<ds>_boxes.jpg` |
| `nabirds_cases.py` | 分析 | 找"鸟占图 < 10%、原图预测错、裁剪后预测对"的例子，画原图+框 / 裁剪图，标类别名 | → `/tmp/nabirds_cases.jpg` |

## 最小流程（以 NABirds 为例）

```bash
export HF_ENDPOINT=https://hf-mirror.com                     # DINOv3 权重走镜像
python locate/make_crops.py --dataset nabirds --tau 0.07 --margin 0.15 --square --suffix _sq
for v in raw dinov3_sq gt_sq; do
  python locate/train_vit.py --data_root data/crops/nabirds/$v --name nabirds_$v
done
python locate/analyze_size.py nabirds data/crops/nabirds/boxes_sq.csv output nabirds_raw nabirds_dinov3_sq nabirds_gt_sq
```

## 几个设计上的决定（为什么是这样）

- **框要松，不要紧**：紧贴主体的框在 CUB 上掉 0.5～2.5 个点（细长框被 `Resize((600,600))` 拉变形 + 丢上下文）。`--square --margin 0.15` 是校准出来的最终配置，官方框也做同样处理，保证公平。
- **裁剪是放大，不是涂灰**：前置实验（`experiments/notes/03`）证明冻结 ViT 涂掉背景不涨，收益来自主体像素变多。
- **τ=0.07**：`--calib 1000` 在 CUB 训练集上对官方 bbox 扫出来的（mean IoU 0.79），ncut/TokenCut 式二分只有 0.37，保留在代码里只是为了对照。
- **top-k 固定 k**：token 选择每张图保留同样多的 token，算力可比；随机对照用同一个 k。
- **DINOv3 全程 fp32**：fp16 下注意力 logits 会溢出成 NaN，`dino_attn.py` 内部关掉了 autocast。
- **12GB 显卡**：`--train_batch_size 8 --grad_accum 2`，有效 batch 不变。

## 数据集目录结构

`make_crops.py` 的五个 lister 按各数据集官方发布格式读：CUB / NABirds 用 `images.txt` + `train_test_split.txt` + `bounding_boxes.txt`；Dogs 用 `train_list.mat` / `test_list.mat` + `Annotation/` 里的 XML；Cars 用 `train|test/<类名>/` 目录 + `anno_{train,test}.csv`；IP102 用 `train|test/<label>/` 目录。详见根目录 README §3.2。
