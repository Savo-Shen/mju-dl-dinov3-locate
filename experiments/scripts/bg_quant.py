"""定量验证：冻结 ViT 的 patch 特征，前景分离质量和背景杂乱程度的关系。

回答两个问题：
  1. DINOv3 的密集特征是否真的比 DINOv2 更「干净」？干净是 register 的功劳还是 Gram anchoring 的？
  2. 背景越乱，各模型的前景分离掉得多少？v3 是否在杂乱背景上失去优势？

数据：CUB-200-2011（自带每张图的鸟体分割 mask，当前景 ground truth）。
指标（每张图 × 每个模型）：
  iou        NCut（patch 相似度图 Normalized Cut，自适应阈值）前景 mask 与 GT 的 IoU
  iou_pc1    单图 PC1 + Otsu 的 IoU，副指标，只用来看结论对 mask 方法敏不敏感
  smooth     每个 patch 与上下左右邻居的平均余弦相似度（越高越平滑）
  artifact   patch 范数的离群比例（median + 3·MAD 之外），衡量 artifact token
协变量（每张图，与模型无关）：
  clutter    GT 背景区域的平均梯度幅值（越高越乱）
  fg_ratio   鸟占画面比例

用法：
  python experiments/scripts/bg_quant.py --n 600
  python experiments/scripts/bg_quant.py --n 600 --analyze-only   # 只重跑统计
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from scipy import stats
from sklearn.decomposition import PCA
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
CUB = ROOT / "data" / "CUB" / "CUB_200_2011"
SEG = ROOT / "data" / "CUB" / "segmentations"
OUT = ROOT / "outputs" / "bg_quant"

MODELS = {
    "dinov2":      "vit_small_patch14_dinov2.lvd142m",
    "dinov2_reg4": "vit_small_patch14_reg4_dinov2.lvd142m",
    "dinov3":      "vit_small_patch16_dinov3.lvd1689m",
}


# ---------------------------------------------------------------- 数据
def load_cub_list(n, seed, split="test"):
    """从 CUB 官方 split 里抽 n 张。用 test split 是因为主干没在 CUB 上训过，train/test 无所谓，
    但固定用一个方便和后续线性探针实验对齐。"""
    ids = pd.read_csv(CUB / "images.txt", sep=" ", names=["id", "path"])
    sp = pd.read_csv(CUB / "train_test_split.txt", sep=" ", names=["id", "is_train"])
    df = ids.merge(sp, on="id")
    df = df[df.is_train == (1 if split == "train" else 0)]
    rng = random.Random(seed)
    rows = df.sample(n=min(n, len(df)), random_state=seed).sort_values("id")
    return rows.path.tolist()


def gt_mask(rel_path, grid):
    """CUB 的分割图是灰度 png，鸟体 255；多标注者的图有中间值，取 >128。缩到 patch 网格。"""
    p = SEG / rel_path.replace(".jpg", ".png")
    m = Image.open(p).convert("L").resize((grid, grid), Image.BILINEAR)
    return np.asarray(m) > 128


def clutter_and_fg(rel_path, size=224):
    """背景杂乱度 = GT 背景区域的平均梯度幅值。用固定 224 分辨率算，和模型无关。"""
    img = np.asarray(Image.open(CUB / "images" / rel_path).convert("L")
                     .resize((size, size), Image.BILINEAR), dtype=np.float32) / 255
    seg = np.asarray(Image.open(SEG / rel_path.replace(".jpg", ".png")).convert("L")
                     .resize((size, size), Image.BILINEAR)) > 128
    gy, gx = np.gradient(img)
    mag = np.hypot(gx, gy)
    bg = ~seg
    clutter = float(mag[bg].mean()) if bg.any() else float("nan")
    return clutter, float(seg.mean())


# ---------------------------------------------------------------- 特征与指标
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from vitfeat import (artifact_ratio, iou, ncut_mask, pc1_mask,  # noqa: E402
                     smoothness)


@torch.no_grad()
def run_model(key, name, paths, grid, device, bs):
    model = timm.create_model(name, pretrained=True, num_classes=0,
                              dynamic_img_size=True).eval().to(device)
    patch = model.patch_embed.patch_size[0]
    size = grid * patch
    cfg = timm.data.resolve_model_data_config(model)
    tf = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])
    rows = []
    for i in range(0, len(paths), bs):
        chunk = paths[i:i + bs]
        x = torch.stack([tf(Image.open(CUB / "images" / p).convert("RGB")) for p in chunk]).to(device)
        f = model.forward_features(x)[:, model.num_prefix_tokens:, :]
        f = f.reshape(len(chunk), grid, grid, -1).float().cpu().numpy()
        for p, feat in zip(chunk, f):
            gt = gt_mask(p, grid)
            rows.append(dict(
                path=p, model=key,
                iou=iou(ncut_mask(feat), gt),        # 主指标：NCut（自适应阈值）
                iou_pc1=iou(pc1_mask(feat), gt),     # 副指标：单图 PC1+Otsu，只为看方法敏感性
                smooth=smoothness(feat),
                artifact=artifact_ratio(feat),
            ))
        print(f"  [{key}] {min(i + bs, len(paths))}/{len(paths)}", end="\r", flush=True)
    print(f"  [{key}] done · 输入 {size}px · patch{patch}          ")
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return rows


# ---------------------------------------------------------------- 统计
def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x); x = x[~np.isnan(x)]
    means = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return float(np.mean(x)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def analyze(df):
    wide = df.pivot(index="path", columns="model", values=["iou", "iou_pc1", "smooth", "artifact"])
    cov = df.drop_duplicates("path").set_index("path")[["clutter", "fg_ratio"]]
    lines = []
    P = lines.append

    P("## 总体（均值 [95% bootstrap CI]）\n")
    P("> artifact 列**不可用**：绝对阈值（范数 > 2× 中位数）下三个模型都是 0%，ViT-S 没有高范数 "
      "artifact token；这里的差异是相对阈值（median+3·MAD）在紧分布上的误报。保留只为可追溯。\n")
    P("| 模型 | IoU (NCut) ↑ | IoU (PC1) ↑ | 平滑度 ↑ | artifact 比例 ↓ |")
    P("|---|---|---|---|---|")
    for m in MODELS:
        cells = []
        for k in ["iou", "iou_pc1", "smooth", "artifact"]:
            mu, lo, hi = boot_ci(wide[k][m])
            cells.append(f"{mu:.3f} [{lo:.3f}, {hi:.3f}]")
        P(f"| {m} | " + " | ".join(cells) + " |")

    P("\n## 配对检验（Wilcoxon，同一批图）\n")
    P("| 对比 | 指标 | 差值均值 | p 值 | 更好的一方 |")
    P("|---|---|---|---|---|")
    for a, b in [("dinov2", "dinov2_reg4"), ("dinov2_reg4", "dinov3"), ("dinov2", "dinov3")]:
        for k, better_high in [("iou", True), ("smooth", True), ("artifact", False)]:
            xa, xb = wide[k][a], wide[k][b]
            ok = ~(xa.isna() | xb.isna())
            d = (xb[ok] - xa[ok])
            p = stats.wilcoxon(xa[ok], xb[ok]).pvalue if d.abs().sum() > 0 else 1.0
            winner = b if (d.mean() > 0) == better_high else a
            P(f"| {a} → {b} | {k} | {d.mean():+.4f} | {p:.2e} | {winner} |")

    P("\n## 按背景杂乱度分层（三等分）\n")
    q = pd.qcut(cov.clutter, 3, labels=["低", "中", "高"])
    P("| 杂乱度 | n | " + " | ".join(f"{m} IoU" for m in MODELS) + " | v3 − v2 | v3 − v2reg4 |")
    P("|---|---|" + "---|" * (len(MODELS) + 2))
    for lvl in ["低", "中", "高"]:
        idx = q[q == lvl].index
        sub = wide["iou"].loc[idx]
        cells = [f"{sub[m].mean():.3f}" for m in MODELS]
        d1 = (sub["dinov3"] - sub["dinov2"]).mean()
        d2 = (sub["dinov3"] - sub["dinov2_reg4"]).mean()
        P(f"| {lvl} | {len(idx)} | " + " | ".join(cells) + f" | {d1:+.3f} | {d2:+.3f} |")

    P("\n## IoU 与杂乱度的相关（Spearman）\n")
    P("| 模型 | ρ | p |")
    P("|---|---|---|")
    for m in MODELS:
        ok = ~wide["iou"][m].isna()
        r, p = stats.spearmanr(cov.clutter[ok], wide["iou"][m][ok])
        P(f"| {m} | {r:+.3f} | {p:.2e} |")

    P("\n## 双重分层：v3 − v2 的 IoU 优势（行=鸟体大小，列=杂乱度）\n")
    q_fg = pd.qcut(cov.fg_ratio, 3, labels=["小鸟", "中鸟", "大鸟"])
    P("| | 低杂乱 | 中杂乱 | 高杂乱 |")
    P("|---|---|---|---|")
    for a in ["小鸟", "中鸟", "大鸟"]:
        cells = []
        for b in ["低", "中", "高"]:
            idx = cov.index[(q_fg == a) & (q == b)]
            d = wide["iou"].loc[idx, "dinov3"] - wide["iou"].loc[idx, "dinov2"]
            cells.append(f"{d.mean():+.3f} (n={len(idx)})")
        P(f"| {a} | " + " | ".join(cells) + " |")

    P("\n## 控制鸟体大小后，IoU 与杂乱度的偏相关（对 fg_ratio 回归取残差再算 Spearman）\n")
    P("| 模型 | ρ_partial | p |")
    P("|---|---|---|")
    for m in MODELS:
        y = wide["iou"][m].values; x1 = cov.clutter.values; x2 = cov.fg_ratio.values
        ok = ~np.isnan(y)
        ry = y[ok] - np.polyval(np.polyfit(x2[ok], y[ok], 1), x2[ok])
        rx = x1[ok] - np.polyval(np.polyfit(x2[ok], x1[ok], 1), x2[ok])
        rr, pp = stats.spearmanr(rx, ry)
        P(f"| {m} | {rr:+.3f} | {pp:.2e} |")

    P("\n## 按鸟体大小分层（三等分）\n")
    q2 = pd.qcut(cov.fg_ratio, 3, labels=["小", "中", "大"])
    P("| 鸟体占比 | n | " + " | ".join(f"{m} IoU" for m in MODELS) + " |")
    P("|---|---|" + "---|" * len(MODELS))
    for lvl in ["小", "中", "大"]:
        idx = q2[q2 == lvl].index
        sub = wide["iou"].loc[idx]
        P(f"| {lvl} | {len(idx)} | " + " | ".join(f"{sub[m].mean():.3f}" for m in MODELS) + " |")

    return "\n".join(lines), wide, cov


def plot(wide, cov, out):
    import matplotlib
    matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "PingFang HK", "Heiti SC", "Arial Unicode MS"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    colors = {"dinov2": "#888", "dinov2_reg4": "#4a90d9", "dinov3": "#d9534f"}

    # 1. IoU vs clutter，分箱后画均值
    ax = axes[0]
    bins = pd.qcut(cov.clutter, 8, labels=False)
    for m in MODELS:
        g = wide["iou"][m].groupby(bins)
        ax.plot(cov.clutter.groupby(bins).mean(), g.mean(), "o-", label=m, color=colors[m])
    ax.set_xlabel("背景杂乱度（背景区域平均梯度）"); ax.set_ylabel("前景 IoU")
    ax.set_title("背景越乱，前景分离越差？"); ax.legend(fontsize=8)

    # 2. 平滑度分布
    ax = axes[1]
    for m in MODELS:
        ax.hist(wide["smooth"][m].dropna(), bins=40, alpha=0.5, label=m, color=colors[m])
    ax.set_xlabel("patch 邻域余弦相似度"); ax.set_title("特征平滑度"); ax.legend(fontsize=8)

    # 3. v3 − v2 优势：鸟体大小 × 杂乱度 热力图
    ax = axes[2]
    q_fg = pd.qcut(cov.fg_ratio, 3, labels=["小鸟", "中鸟", "大鸟"])
    q_cl = pd.qcut(cov.clutter, 3, labels=["低", "中", "高"])
    grid = np.zeros((3, 3))
    for i, a in enumerate(["小鸟", "中鸟", "大鸟"]):
        for j, b in enumerate(["低", "中", "高"]):
            idx = cov.index[(q_fg == a) & (q_cl == b)]
            grid[i, j] = (wide["iou"].loc[idx, "dinov3"] - wide["iou"].loc[idx, "dinov2"]).mean()
    im = ax.imshow(grid, cmap="Reds", vmin=0, vmax=max(0.25, grid.max()))
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{grid[i, j]:+.3f}", ha="center", va="center",
                    color="white" if grid[i, j] > 0.13 else "black", fontsize=10)
    ax.set_xticks(range(3)); ax.set_xticklabels(["低", "中", "高"])
    ax.set_yticks(range(3)); ax.set_yticklabels(["小鸟", "中鸟", "大鸟"])
    ax.set_xlabel("背景杂乱度"); ax.set_title("DINOv3 − DINOv2 的 IoU 优势")
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"图已保存 {out}")


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    csv = OUT / f"per_image_n{args.n}_seed{args.seed}.csv"

    if not args.analyze_only:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        paths = load_cub_list(args.n, args.seed)
        print(f"设备 {device} · {len(paths)} 张 · 网格 {args.grid}x{args.grid}")

        print("算协变量 ...")
        cov = {p: clutter_and_fg(p) for p in paths}

        rows = []
        for key, name in MODELS.items():
            rows += run_model(key, name, paths, args.grid, device, args.bs)
        df = pd.DataFrame(rows)
        df["clutter"] = df.path.map(lambda p: cov[p][0])
        df["fg_ratio"] = df.path.map(lambda p: cov[p][1])
        df.to_csv(csv, index=False)
        print(f"逐图结果 {csv}")
    else:
        df = pd.read_csv(csv)

    report, wide, cov = analyze(df)
    md = OUT / f"report_n{args.n}_seed{args.seed}.md"
    md.write_text(f"# 背景依赖定量结果 · n={args.n} · seed={args.seed} · grid={args.grid}\n\n{report}\n")
    print(report)
    plot(wide, cov, OUT / f"plot_n{args.n}_seed{args.seed}.png")
    print(f"报告 {md}")


if __name__ == "__main__":
    main()
