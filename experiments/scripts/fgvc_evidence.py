"""实验 04 · 线性探针认这只鸟时，证据在哪？（patch 级精确分解）

线性探针在 mean-pooled patch 特征上是线性的：
    logit_c = Σ_d w_cd · (mean_d − μ_d) / σ_d + b_c
            = (1/N) Σ_i [ Σ_d w_cd · (patch_id − μ_d) / σ_d ] + b_c
所以第 i 个 patch 对类别 c 的贡献 s_i = Σ_d w_cd (patch_id − μ_d)/σ_d 是**精确**的，不是 attention 近似。

指标（每张图 × 每个模型，对真实类别）：
  evidence_in   正贡献落在 GT 鸟体 mask 内的比例
  mask_area     mask 占 patch 网格的比例（如果证据均匀分布，evidence_in 就等于它）
  concentration evidence_in / mask_area，>1 说明证据向鸟集中
另存若干张证据热力图，看模型「认鸟看哪里」。

前置：先跑 fgvc_bg.py extract 和 fgvc_bg.py probe --kind mean
用法：
  python experiments/scripts/fgvc_evidence.py --n 500 --viz 8
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import stats
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vitfeat as vf  # noqa: E402
from fgvc_bg import CUB, SEG, OUT, load_meta  # noqa: E402

EV = OUT / "evidence"


@torch.no_grad()
def patch_tokens(model, img, grid, cfg, device):
    patch = model.patch_embed.patch_size[0]
    size = grid * patch
    tf = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), transforms.Normalize(cfg["mean"], cfg["std"]),
    ])
    x = tf(img.convert("RGB")).unsqueeze(0).to(device)
    f = model.forward_features(x)[0, model.num_prefix_tokens:].float().cpu().numpy()
    return f.reshape(grid, grid, -1)


def evidence_map(tokens, probe, label):
    """每个 patch 对真实类别 logit 的贡献。"""
    sc, clf = probe["scaler"], probe["clf"]
    w = clf.coef_[label] / sc.scale_               # (D,)
    mu = sc.mean_
    return (tokens - mu) @ w                        # (g, g)


def gt_mask(rel_path, grid):
    m = Image.open(SEG / rel_path.replace(".jpg", ".png")).convert("L").resize((grid, grid), Image.BILINEAR)
    return np.asarray(m) > 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--viz", type=int, default=8, help="存几张热力图")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    device = vf.pick_device()
    df = load_meta()
    test = df[df.is_train == 0].sample(n=a.n, random_state=a.seed).sort_values("id")
    EV.mkdir(parents=True, exist_ok=True)

    import timm
    rows, viz = [], {}
    for k in vf.MODELS:
        probe = joblib.load(OUT / "probes" / f"{k}__mean__C{a.C}.joblib")
        model = vf.load_model(k, device)
        cfg = timm.data.resolve_model_data_config(model)
        for j, (_, r) in enumerate(test.iterrows()):
            img = Image.open(CUB / "images" / r.path)
            tok = patch_tokens(model, img, a.grid, cfg, device)
            s = evidence_map(tok, probe, int(r.label))
            gt = gt_mask(r.path, a.grid)
            pos = np.clip(s, 0, None)
            ev_in = float(pos[gt].sum() / (pos.sum() + 1e-8))
            area = float(gt.mean())
            # 预测对不对（用 mean 特征）
            pred = probe["clf"].predict(probe["scaler"].transform(tok.reshape(-1, tok.shape[-1]).mean(0, keepdims=True)))[0]
            rows.append(dict(model=k, path=r.path, evidence_in=ev_in, mask_area=area,
                             concentration=ev_in / (area + 1e-8), correct=int(pred == r.label)))
            if j < a.viz:
                viz.setdefault(r.path, {})[k] = (s, gt)
            print(f"  [{k}] {j + 1}/{len(test)}", end="\r", flush=True)
        print(f"  [{k}] done                    ")

    out = pd.DataFrame(rows)
    out.to_csv(EV / f"per_image_n{a.n}.csv", index=False)

    # ---- 报告
    L = []
    P = L.append
    P(f"# 证据在哪 · CUB test 随机 {a.n} 张 · grid {a.grid} · 线性探针(mean 特征, C={a.C})\n")
    P("| 模型 | 正证据落在鸟上的比例 | 鸟占画面比例 | 集中度（前者/后者） | 探针准确率（本子集） |")
    P("|---|---|---|---|---|")
    for k in vf.MODELS:
        g = out[out.model == k]
        P(f"| {k} | {g.evidence_in.mean():.3f} | {g.mask_area.mean():.3f} | "
          f"{g.concentration.mean():.2f}× | {g.correct.mean():.3f} |")
    P("\n配对 Wilcoxon（evidence_in）：")
    w = out.pivot(index="path", columns="model", values="evidence_in")
    for a_, b_ in [("dinov2", "dinov2_reg4"), ("dinov2_reg4", "dinov3"), ("dinov2", "dinov3")]:
        d = w[b_] - w[a_]
        P(f"- {a_} → {b_}: {d.mean():+.3f}, p = {stats.wilcoxon(w[a_], w[b_]).pvalue:.1e}")
    P("\n分对 vs 分错的图，证据集中度：")
    for k in vf.MODELS:
        g = out[out.model == k]
        P(f"- {k}: 分对 {g[g.correct == 1].concentration.mean():.2f}× · 分错 {g[g.correct == 0].concentration.mean():.2f}×")
    P("\n## 怎么读\n")
    P("- 集中度 = 1 表示正证据均匀撒在全图；越大越集中在鸟上。")
    P("- 注意这个比例受「背景和真实类别是否一致」影响：栖息地典型（海鸟在海上）时背景也给真实类别加分，")
    P("  比例被拉低；栖息地反常时正证据只能来自鸟，比例升高。所以**分错的图集中度反而更高**——")
    P("  不是「看错地方」，而是「背景没帮忙」。这个指标衡量的是**背景对分类贡献了多少**，不是注意力对不对。")
    (EV / f"report_n{a.n}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    # ---- 热力图
    import matplotlib
    matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "PingFang HK", "Heiti SC", "Arial Unicode MS"]
    import matplotlib.pyplot as plt
    paths = list(viz)
    fig, axes = plt.subplots(len(paths), 1 + len(vf.MODELS), figsize=(2.4 * (1 + len(vf.MODELS)), 2.4 * len(paths)), squeeze=False)
    for i, p in enumerate(paths):
        img = Image.open(CUB / "images" / p).convert("RGB").resize((256, 256))
        axes[i][0].imshow(img); axes[i][0].set_title(p.split("/")[0][4:][:16], fontsize=7)
        for j, k in enumerate(vf.MODELS, 1):
            s, gt = viz[p][k]
            pos = np.clip(s, 0, None)
            axes[i][j].imshow(img, alpha=0.45)
            axes[i][j].imshow(np.kron(pos, np.ones((256 // a.grid, 256 // a.grid))), cmap="hot", alpha=0.6)
            axes[i][j].contour(np.kron(gt, np.ones((256 // a.grid, 256 // a.grid))), levels=[0.5], colors="cyan", linewidths=0.8)
            ev = float(pos[gt].sum() / (pos.sum() + 1e-8))
            axes[i][j].set_title(f"{k} · 证据在鸟 {ev:.0%}", fontsize=7)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(EV / f"heatmaps_n{a.n}.png", dpi=150)
    print(f"热力图 → {EV / f'heatmaps_n{a.n}.png'}")


if __name__ == "__main__":
    main()
