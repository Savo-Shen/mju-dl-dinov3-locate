# 实验 01 · DINOv2 vs DINOv3 的 patch 特征可视化

日期：2026-09-10 · 设备：M1 Max / MPS · 单次运行约 20 秒

## 做法

- 模型：`vit_small_patch14_dinov2.lvd142m`、`vit_small_patch16_dinov3.lvd1689m`（都是 22M 参数）
- **按 patch 网格对齐而不是按像素**：目标 32×32 网格，v2(p14) 输入 448px，v3(p16) 输入 512px。
  直接用同一个像素尺寸是不公平的——两者 patch size 不同，网格数会差一倍。
- 所有图共用一套 PCA，颜色才可比。
- 前景 mask 用第一主成分 + **Otsu 阈值**。
  一开始用的是中位数，等于强行规定一半 patch 是前景，主体小的时候必然错，换掉了。

```bash
python experiments/scripts/pca_features.py --images experiments/data --grid 32 --mask \
  --out experiments/outputs/pca_v2_vs_v3.jpg
```

## 观察

1. **DINOv3 的特征图明显更平滑**。DINOv2 背景里有大量椒盐状高频噪点（artifact token），
   DINOv3 基本没有。这一条支持论文关于 Gram anchoring 的说法。
2. **中等复杂度背景上 DINOv3 前景分离更干净**。蓝黑雀那张，DINOv3 的 mask 是一个干净的
   鸟形轮廓，DINOv2 的 mask 在蕨类叶子上散了一片噪点。
3. **但最杂乱的那张（黄嘴杜鹃）DINOv3 没有优势**，mask 把大片背景也算进了前景。

## 结论与下一步

「dense feature 更锐利」不等于「更不依赖背景」——第 3 条就是反例。
定性图只能看个大概，需要定量：用 SAM 抠主体做 原图/背景涂灰/背景替换 三组变体，
测冻结特征 + 线性探针的掉分幅度，才能回答哪个主干真的更看主体。

## 已知局限

- 只看了 4 张图，small 尺寸，不能外推。
- Otsu 仍是全局单阈值，主体极小时会偏。
- PCA 前三维只解释了一部分方差，没统计解释率。
