# DINOv3 课堂汇报 · 项目索引

> 一份回溯用的总索引：课程要求 → 汇报结构 → 每一页用到的代码 / 数据 / 图 / 数字在哪里。
> 整理日期 2026-09-17。汇报 PPT：`~/Projects/Slidev-Projects/DINOv3-Talk/`（iCloud 里 `课程/深度学习/分享/DINOv3-Talk` 是它的替身）。

---

## 1. 课程要求（来自 `课程/深度学习/深度学习.pdf`）

| 项目 | 要求 |
|---|---|
| 形式 | 开源项目**轻量复现**，不是"介绍一个仓库"：发现项目 → 读懂原理 → 跑通代码 → 展示价值 |
| 时长 | 12 分钟 = **6 分钟讲解 + 4 分钟 Demo + 2 分钟问答**，每次课前 2 组 |
| 提交 | 上课前一天交 GitHub 链接 + PPT |
| 选题标准 | 可运行、可解释、和课程知识点有明确连接（DINOv3 对应 **ViT / 视觉预训练** 这条线） |
| 必答 7 问 | ① 解决什么问题 ② 为什么值得关注 ③ 核心原理 ④ 如何安装运行 ⑤ 实际复现结果 ⑥ 对应课程知识点 ⑦ 局限与改进 |
| 评分 | 课堂汇报占 20%；小组分 × 个人贡献系数（Git commit / 文档 / 实验记录 / 组内互评 / 现场追问） |
| 底线 | 必须有代码、结果、分析和可展示成果；只有 PPT / 只调 API / 无 README 不可复现 = 红线 |
| 延展 | 汇报选题建议发展为期末项目 baseline（期末实践课 12/18 设计与架构、12/25 评测与结果） |

**7 问在 PPT 里的位置**：①② → 第 3–4 页；③ → 第 5 页；④ → 第 7 页；⑤ → 第 8–14 页；⑥ → 第 5 页（ViT 结构、自监督）+ 第 14 页（token/注意力）；⑦ → 第 15 页。

---

## 2. 汇报对象

| | |
|---|---|
| 仓库 | https://github.com/facebookresearch/dinov3 · Meta FAIR · 2025-08-14 发布 |
| 数据（2026-09-17 取） | ⭐ 11.4k · Fork 960 · 70 commits · 12 contributors · 145 issues · 20 PR |
| 复现固定的 commit | `6876159`（Remove unused LinearW24 #352，也是当前 HEAD） |
| 论文 | DINOv3 https://arxiv.org/abs/2508.10104 · DINO https://arxiv.org/abs/2104.14294 · ViT https://arxiv.org/abs/2010.11929 |
| 本机用的权重 | timm `vit_small_patch16_dinov3.lvd1689m`（21M，Demo / 前景分离实验）；服务器用 `vit_base_patch16_dinov3.lvd1689m`（找框 / 选 token） |
| 权重镜像 | `HF_ENDPOINT=https://hf-mirror.com`（github.com 直连不稳） |

---

## 3. 代码与数据在哪

### 3.1 Mac（本机复现 + Demo）— `~/Projects/dinov3/`

上游仓库原样，所有工作在 `experiments/`：

| 路径 | 内容 |
|---|---|
| `experiments/app/run.sh` → http://127.0.0.1:8000 | Gradio/自建 Demo：单图三模型（dinov2 / dinov2_reg4 / dinov3）并排，PCA 着色 + 前景 mask，M1 Max MPS 约 1.5 s/图 |
| `experiments/app/server.py`, `static/` | Demo 后端与页面 |
| `experiments/data/CUB/CUB_200_2011/` | CUB 全量（含 `segmentations/` 官方前景 mask） |
| `experiments/notes/01-pca-v2-vs-v3.md` | 实验 01：PCA 可视化，DINOv2 vs v3（PPT 第 3 页图） |
| `experiments/notes/02-bg-quant-n1000.md` | 实验 02：1000 张前景 IoU / 平滑度（PPT 第 9 页） |
| `experiments/notes/03-fgvc-background.md` | 实验 03：冻结 ViT + 线性探针，原图 / 涂背景 / 涂鸟（备份数据：三代都 84%，涂掉背景不涨，mean-pool 时涂背景 +12） |
| `experiments/outputs/` | 各实验的表、曲线、可视化、失败记录 |
| conda 环境 | `dinov3`（`~/miniforge3/envs/dinov3`，画 PPT 图也用它的 matplotlib） |

### 3.2 lab 服务器（主实验）— `ssh lab` / `ssh lab-cloudflare`，`~/Programs/dinov3-crop-vit/`

conda 环境 `fal-wyh`（torch 1.13 + timm 1.0.25）。GPU0 是同学 lsj 的，借用前看 `nvidia-smi`；GPU1 自己用。

| 文件 | 作用 |
|---|---|
| `make_crops.py` | **阶段 1**：冻结 DINOv3 ViT-B/16 最后一层 CLS→patch 注意力（28×28）→ 按 max 归一化、τ=0.07 二值化 → 注意力质量最大的连通块外接框 → 裁剪。`--square --margin 0.15 --suffix _sq` 是最终用的变体（补成正方形 + 外扩 15%）；`--calib N` 对官方 bbox 扫参；`--img 224` 找框只看 224 |
| `train_vit.py` | **阶段 2**：纯 ViT-B/16（ImageNet-21k npz）在 ImageFolder 上微调。600→RandomCrop 448 + hflip，SGD 3e-2，warmup 500 + cosine，10k 步，batch 16，fp16。`--img_size 224` 自动按比例缩放数据管线。输出 `output/<name>.json`（best / 每千步曲线）和 `output/<name>_preds.npz`（逐图预测） |
| `train_vit_tok.py` | **token 选择变体**：patch embedding 后按 DINOv3 注意力保留 top-k token（CLS 永远保留），`--select dinov3|random|none --keep 0.25 --img_size 896`。DINOv3 在线打分、固定看 448 |
| `analyze_size.py <ds> boxes.csv output name…` | 按官方 bbox 面积占比分 5 组算各 run 的准确率（PPT 第 12 页数据） |
| `viz_boxes.py <ds>` | 红框（自动）/ 青框（官方）抽样图 → `/tmp/<ds>_boxes.jpg`（PPT 第 10 页图） |
| `nabirds_cases.py` | 挑"鸟 < 10%、原图错、裁剪对"的例子 → `/tmp/nabirds_cases.jpg`（PPT 第 13 页图） |
| `queue_*.sh` | 各批次队列脚本（按时间：`queue_crops` → `queue_train`（紧框）→ `queue_train_sq` → `queue_224_probe` / `queue_224b` → `queue_s1` → `queue_tok`）；`logs/tok_decision.log` 记录 token 实验的自动决策 |
| `logs/train_<name>.log` | 每个 run 的完整日志 |
| `output/` | 全部 run 的 json + preds（清单见 §5） |

依赖：`third_party/vit_pytorch/`（jeonsworld 纯 ViT 实现）和 `weights/imagenet21k_ViT-B_16.npz`（ImageNet-21k 预训练权重）。

**数据**：原始数据集在 `/data/`（只读）：`CUB_200_2011`、`Stanford_Dogs`、`Stanford_Cars`、`nabirds`、`IP102/classification`。
裁剪数据在 `~/data_crops/<ds>/`（43 GB）：
- `raw/`（软链接原图）、`dinov3/` `gt/`（第一版紧框，已弃用）、`dinov3_sq/` `gt_sq/`（最终版）、cub 另有 `dinov3_sq224/`
- 全部是 ImageFolder 布局 `{train,test}/<cls>/<file>`，`train_vit.py --data_root` 直接指向
- `boxes.csv` / `boxes_sq.csv`：每张图的自动框、官方框（有则）、IoU、原图尺寸

### 3.3 savo-lab（RTX 3060，Windows 原生）— `ssh savo-lab`，`D:\dinov3-crop-vit\`

只跑了 CUB：`cub_gt`（紧框）、`cub_dinov3_sq`、`cub_gt_sq`、`cub_raw_s1`、`cub_dinov3_sq_s1`。结果在 `D:\dinov3-crop-vit\output\`，日志 `logs\`。
注意事项：后台任务要用 WMI `Win32_Process.Create` 起（ssh 断开会杀进程，`schtasks` 无人登录时不跑）；机器会偶发卡死重启；与 lab 之间只有 0.5 MB/s 的公网路，大数据从 Mac 走局域网推。

### 3.4 PPT — `~/Projects/Slidev-Projects/DINOv3-Talk/`

`slides.md`（含讲者备注和每页建议秒数；仓库里的 PDF 不含课题组相关的备份页）、`slides-export.pdf`、`public/` 图片。`npm run dev` 预览，`npx slidev export --format pdf --scale 2 --output slides-export.pdf` 导出。

---

## 4. 每一页对应的材料

| 页 | 标题 | 图 / 数据来源 |
|---|---|---|
| 1 | 封面 | — |
| 2 | 项目介绍 | `public/repo_home_crop.png`（`仓库首页.png` 裁剪）；stars 等数字取自 GitHub 2026-09-17 |
| 3 | DINOv3 是用来做什么的 | `pca_v2_vs_v3.png` ← Mac 实验 01 |
| 4 | 项目背景 | 文字 |
| 5 | 基本原理 | 文字（ViT / DINO 学生-老师 / Gram anchoring） |
| 6 | 我们做了什么 | 文字（三步：Mac 跑通 → 1000 张定量 → 先找主体再分类） |
| 7 | 安装与运行 | 代码；三模型表 |
| 8 | 运行效果 | `evidence_heatmaps.png` ← Mac Demo |
| 9 | 前景分离 | `bg_quant_n1000.png` ← Mac 实验 02（IoU 0.681 vs 0.560，9/9 组领先，平滑度 0.949） |
| 10 | 实验设计 | `nabirds_boxes.jpg` ← lab `viz_boxes.py nabirds`；IoU 表 ← `logs/crops_*.log` |
| 11 | 方向一：先找再放大 | `crop_results_bar.png` ← §5 表 A |
| 12 | NABirds 分析 | `nabirds_size_bins.png` ← `analyze_size.py nabirds`；224 结果 ← §5 表 C |
| 13 | NABirds 案例 | `nabirds_cases.jpg` ← lab `nabirds_cases.py` |
| 14 | 方向二：只看 1/4 token | `token_select.png` ← §5 表 D |
| 15 | 局限与改进 | 紧框结果 ← §5 表 B |
| 16 | Demo | Mac `experiments/app/run.sh` |
| 17 | Q&A | 备注里有 7 条预判问题 |
| 18 | Backup：复现材料 | — |

画图脚本：第 11 / 12 / 14 页和 224 图是在 Mac 上用 `~/miniforge3/envs/dinov3/bin/python` + matplotlib 画的，数据直接写在脚本里（数字见 §5），需要重画时按 §5 的表填即可。

---

## 5. 全部实验结果（Top-1 %，ViT-B/16，seed 42；`_s1` = seed 1）

### A. 主实验：三种输入（`_sq` = 正方形框 + 15% 上下文，最终版）

| 数据集 | raw 原图 | DINOv3 自动框裁剪 | 官方 bbox 裁剪 | 自动框 IoU（≥0.5 占比） |
|---|---|---|---|---|
| CUB | 90.68 / 90.61 (s1) | 90.58 / 90.82 (s1) | 90.87 | 0.796（97%） |
| Dogs | 90.43 / 90.94 (s1) | 90.29 / 90.50 (s1) | 90.47 | 0.767（90%） |
| Cars | 88.73 / 88.60 (s1) | 88.91 / 88.75 (s1) | 89.07 | 0.801（98%） |
| **NABirds** | 89.77 / 89.79 (s1) | **91.14 / 91.14 (s1)** | 91.05 | 0.789（96%） |
| IP102 | 73.38 | 73.41 | —（无 bbox） | — |

NABirds 按鸟占图面积分 5 组（每组 ≈ 4927 张）：

| 面积占比 | <15% | 15–24% | 24–33% | 33–45% | >45% |
|---|---|---|---|---|---|
| raw | 84.66 | 91.49 | 90.87 | 91.82 | 90.03 |
| DINOv3 裁剪 | **88.33 (+3.7)** | 92.22 | 92.04 | 92.25 | 90.85 |
| 官方框裁剪 | 88.74 | 92.04 | 91.70 | 92.14 | 90.62 |

### B. 第一版紧框（已弃用，PPT 第 15 页"踩过的坑"）

| | raw | DINOv3 紧框 | 官方紧框 |
|---|---|---|---|
| CUB | 90.68 | 90.20 | 88.14 |
| Dogs | 90.43 | 90.06 | 88.82 |
| Cars | 88.73 | 88.47 | — |

原因：细长框被 `Resize((600,600))` 拉变形 + 丢背景上下文；越紧掉越多。

### C. 分辨率探针（CUB，输入 224）

| raw @224 | DINOv3 裁剪 @224（找框在 448） | 官方框裁剪 @224 | DINOv3 找框也在 224 | raw @448 |
|---|---|---|---|---|
| 88.44 | **90.35 (+1.9)** | 89.61 | 89.30 | 90.68 |

### D. token 选择（ViT 只看部分 patch）

| 保留 token | 谁选 | CUB @448 | NABirds @448 |
|---|---|---|---|
| 100% | — | 90.68 | 89.77 |
| 50% | DINOv3 | 90.63 | — |
| 25% | DINOv3 | 90.09 | 89.40 |
| 10% | DINOv3 | 88.73 | 87.63 |
| 25% | **随机** | **83.69** | **76.43** |
| 25%，输入 896 | DINOv3 | 90.27 | 90.51 |

### E. Mac 上的前置实验（冻结特征，ViT-S）

- 前景分离 1000 张：DINOv3 IoU 0.681 vs DINOv2 0.560（+0.121），9 个"主体大小 × 背景复杂度"分组全部领先；平滑度 0.949；DINOv2 + register 平滑度更高但 IoU 略降。
- 线性探针（CUB）：原图三代都 84%；只剩鸟 73→76；只剩背景 ≈ 10%；**去掉 CLS 用 mean-pool 时涂掉背景反涨 12 点** → CLS 注意力本身就是定位器。

### G. 算力参考（推理 GMACs，ViT-B/16）

224 全图 ≈ 17.6；448 全图 ≈ 78；DINOv3 ViT-B 找框 @448 ≈ 78、@224 ≈ 17.6。

---

## 6. 定稿叙事（讲的时候的主线）

1. DINOv3 不用标注就能把主体框到人工水平（IoU 0.77–0.80，自动框裁剪与官方框裁剪差 ≤ 0.2）。
2. **方向一 · 主体看不清 → 先找再放大换精度**：主体大的 CUB / Dogs / Cars / IP102 持平（±0.4，两个 seed 内），主体小的 NABirds +1.4（两个 seed 都是），最小 20% 的图 +3.7；同一数据集降到 224 也 +1.9。
3. **方向二 · 主体看得清 → 只看 DINOv3 挑的 1/4 token 换算力**：25% token 只掉 0.4–0.6，随机挑同样 25% 掉 7–13。
4. 踩过的坑：紧框掉分（拉变形 + 丢上下文），补成正方形 + 15% 后恢复。
5. 局限：2 个 seed；找框会被更显眼的东西带走（花、卡车、松鼠，NABirds ~4% IoU<0.5）；找框多跑一个 ViT-B。

**措辞提醒**：不要说"裁剪稳定提升"——CUB / Dogs 自动框比原图低 0.1，说"不掉分 + 主体小时显著涨"。

---

## 7. 复现命令速查（lab）

```bash
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate fal-wyh
cd ~/Programs/dinov3-crop-vit && export HF_ENDPOINT=https://hf-mirror.com

# 阶段 1：找框裁剪（最终版）
CUDA_VISIBLE_DEVICES=1 python make_crops.py --dataset nabirds --tau 0.07 --margin 0.15 --square --suffix _sq
# 阶段 2：三种输入各训一次
CUDA_VISIBLE_DEVICES=1 python train_vit.py --data_root ~/data_crops/nabirds/raw       --name nabirds_raw
CUDA_VISIBLE_DEVICES=1 python train_vit.py --data_root ~/data_crops/nabirds/dinov3_sq --name nabirds_dinov3_sq
CUDA_VISIBLE_DEVICES=1 python train_vit.py --data_root ~/data_crops/nabirds/gt_sq     --name nabirds_gt_sq
# 按主体大小分组
python analyze_size.py nabirds ~/data_crops/nabirds/boxes.csv output nabirds_raw nabirds_dinov3_sq nabirds_gt_sq
# token 选择
CUDA_VISIBLE_DEVICES=1 python train_vit_tok.py --data_root ~/data_crops/cub/raw --name cub_tok25 --select dinov3 --keep 0.25
CUDA_VISIBLE_DEVICES=1 python train_vit_tok.py --data_root ~/data_crops/cub/raw --name cub_rand25 --select random --keep 0.25
```

单个 run：448 输入约 1 h（3090 独占）；NABirds / IP102 测试集大，约 1.5 h；224 输入约 15 min。

---

## 8. 时间线

| 日期 | 事件 |
|---|---|
| 09-10 | Mac 上跑通 DINOv3，Demo；实验 01–03 |
| 09-15 | 决定课堂对比改用纯 ViT；写 `make_crops.py` / `train_vit.py`，5 数据集裁剪 + 14 个 run 排队；3060 加入 |
| 09-16 凌晨 | 发现紧框掉分 → 改 `_sq`；全部改跑 _sq |
| 09-16 白天 | _sq 全表出；NABirds +1.4；分组分析；224 探针；PPT 重做 |
| 09-16 晚 | token 选择变体 smoke test（GPU0）→ 自动排完整实验；seed 1 队列 |
| 09-17 | token 全表出；PPT 定稿为"两个方向"；本索引 |
