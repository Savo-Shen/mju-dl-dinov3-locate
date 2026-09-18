"""实验 03 · 细粒度分类的背景依赖（任务层）

问题：冻结的自监督 ViT 做细粒度分类（CUB-200）时，靠的是鸟还是背景？

做法：线性探针在原图训练集上训一次，在四个测试集变体上评：
  orig     原图
  fg       背景涂灰（只剩鸟）          → 掉幅 = 背景对分类的贡献
  bg_mask  鸟体按 mask 涂灰（剪影还在） → 没有鸟还能认出多少
  bg_box   鸟的 bbox 整块涂灰（剪影也没了）→ 更干净的「纯背景」测试
bg_box 准确率远高于随机（1/200 = 0.5%）就说明特征把「物种 ↔ 栖息地」编码进去了。

用法：
  python experiments/scripts/fgvc_bg.py extract            # 抽特征，约 20 分钟
  python experiments/scripts/fgvc_bg.py probe              # 线性探针 + kNN，几分钟
  python experiments/scripts/fgvc_bg.py extract --models dinov3 --grid 16
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vitfeat as vf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CUB = ROOT / "data" / "CUB" / "CUB_200_2011"
SEG = ROOT / "data" / "CUB" / "segmentations"
OUT = ROOT / "outputs" / "fgvc_bg"
FEAT = OUT / "feats"

GRAY = (124, 116, 104)          # ImageNet 均值色，「没有信息」的填充
VARIANTS = ["orig", "fg", "bg_mask", "bg_box"]


# ---------------------------------------------------------------- 数据
def load_meta():
    ids = pd.read_csv(CUB / "images.txt", sep=" ", names=["id", "path"])
    sp = pd.read_csv(CUB / "train_test_split.txt", sep=" ", names=["id", "is_train"])
    lb = pd.read_csv(CUB / "image_class_labels.txt", sep=" ", names=["id", "label"])
    bb = pd.read_csv(CUB / "bounding_boxes.txt", sep=" ", names=["id", "x", "y", "w", "h"])
    df = ids.merge(sp, on="id").merge(lb, on="id").merge(bb, on="id")
    df["label"] -= 1                                    # 0..199
    return df


class CUBVariant(Dataset):
    def __init__(self, df, variant, size, mean, std):
        self.df = df.reset_index(drop=True)
        self.variant = variant
        self.size = size
        self.tf = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(CUB / "images" / r.path).convert("RGB")
        if self.variant != "orig":
            img = self.apply(img, r)
        return self.tf(img), int(r.label)

    def apply(self, img, r):
        arr = np.asarray(img).copy()
        if self.variant == "bg_box":
            x, y, w, h = (int(v) for v in (r.x, r.y, r.w, r.h))
            arr[y:y + h, x:x + w] = GRAY
            return Image.fromarray(arr)
        seg = Image.open(SEG / r.path.replace(".jpg", ".png")).convert("L")
        if seg.size != img.size:
            seg = seg.resize(img.size, Image.BILINEAR)
        m = np.asarray(seg) > 128
        if self.variant == "fg":
            arr[~m] = GRAY
        elif self.variant == "bg_mask":
            arr[m] = GRAY
        return Image.fromarray(arr)


# ---------------------------------------------------------------- 抽特征
@torch.no_grad()
def extract(keys, grid, bs, workers):
    device = vf.pick_device()
    df = load_meta()
    train = df[df.is_train == 1]
    test = df[df.is_train == 0]
    FEAT.mkdir(parents=True, exist_ok=True)

    jobs = [("train", "orig", train), ("train", "fg", train)] + [("test", v, test) for v in VARIANTS]
    for k in keys:
        model = vf.load_model(k, device)
        patch = model.patch_embed.patch_size[0]
        size = grid * patch
        import timm
        cfg = timm.data.resolve_model_data_config(model)
        print(f"[{k}] 输入 {size}px · patch{patch}")

        for split, variant, sub in jobs:
            out = FEAT / f"{k}__{split}__{variant}.npz"
            if out.exists():
                print(f"  跳过（已存在）{out.name}")
                continue
            ds = CUBVariant(sub, variant, size, cfg["mean"], cfg["std"])
            dl = DataLoader(ds, batch_size=bs, num_workers=workers, shuffle=False)
            cls_l, mean_l, y_l = [], [], []
            for i, (x, y) in enumerate(dl):
                f = model.forward_features(x.to(device))
                cls_l.append(f[:, 0].float().cpu())
                mean_l.append(f[:, model.num_prefix_tokens:].mean(1).float().cpu())
                y_l.append(y)
                print(f"  {split}/{variant} {min((i + 1) * bs, len(ds))}/{len(ds)}", end="\r", flush=True)
            np.savez(out, cls=torch.cat(cls_l).numpy(), mean=torch.cat(mean_l).numpy(),
                     y=torch.cat(y_l).numpy(), path=sub.path.values)
            print(f"  {split}/{variant} 完成 → {out.name}            ")
        del model
        if device.type == "mps":
            torch.mps.empty_cache()


# ---------------------------------------------------------------- 探针
def load_feat(k, split, variant, kind):
    z = np.load(FEAT / f"{k}__{split}__{variant}.npz", allow_pickle=True)
    if kind == "cls":
        X = z["cls"]
    elif kind == "mean":
        X = z["mean"]
    else:
        X = np.concatenate([z["cls"], z["mean"]], 1)
    return X, z["y"]


def probe(keys, kind, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    rows = []
    for k in keys:
        Xtr, ytr = load_feat(k, "train", "orig", kind)
        sc = StandardScaler().fit(Xtr)
        Xtr_s = sc.transform(Xtr)
        clf = LogisticRegression(C=C, max_iter=2000).fit(Xtr_s, ytr)
        import joblib
        (OUT / "probes").mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": sc, "clf": clf}, OUT / "probes" / f"{k}__{kind}__C{C}.joblib")
        # kNN 用余弦，特征做 L2 归一化即可
        Xtr_n = Xtr / np.linalg.norm(Xtr, axis=1, keepdims=True)
        knn = KNeighborsClassifier(n_neighbors=20, metric="cosine", weights="distance").fit(Xtr_n, ytr)
        for v in VARIANTS:
            Xte, yte = load_feat(k, "test", v, kind)
            acc_lp = float((clf.predict(sc.transform(Xte)) == yte).mean())
            Xte_n = Xte / np.linalg.norm(Xte, axis=1, keepdims=True)
            acc_knn = float((knn.predict(Xte_n) == yte).mean())
            rows.append(dict(model=k, variant=v, linear=acc_lp, knn=acc_knn))
            print(f"  {k:12s} {v:8s} linear {acc_lp:.3f}  knn {acc_knn:.3f}")
        # 对照：训练集也涂灰背景（fg→fg），排除「灰背景没见过」的分布偏移
        if (FEAT / f"{k}__train__fg.npz").exists():
            Xtr2, ytr2 = load_feat(k, "train", "fg", kind)
            sc2 = StandardScaler().fit(Xtr2)
            clf2 = LogisticRegression(C=C, max_iter=2000).fit(sc2.transform(Xtr2), ytr2)
            Xte, yte = load_feat(k, "test", "fg", kind)
            acc = float((clf2.predict(sc2.transform(Xte)) == yte).mean())
            Xtr2_n = Xtr2 / np.linalg.norm(Xtr2, axis=1, keepdims=True)
            knn2 = KNeighborsClassifier(n_neighbors=20, metric="cosine", weights="distance").fit(Xtr2_n, ytr2)
            acc_k = float((knn2.predict(Xte / np.linalg.norm(Xte, axis=1, keepdims=True)) == yte).mean())
            rows.append(dict(model=k, variant="fg2fg", linear=acc, knn=acc_k))
            print(f"  {k:12s} fg→fg    linear {acc:.3f}  knn {acc_k:.3f}")
    return pd.DataFrame(rows)


def report(df, kind, C):
    OUT.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(df.model))
    L = []
    P = L.append
    P(f"# 细粒度分类的背景依赖 · CUB-200-2011 · 特征={kind} · 线性探针 C={C}\n")
    P("随机基线 = 1/200 = 0.5%\n")
    for metric in ["linear", "knn"]:
        w = df.pivot(index="model", columns="variant", values=metric).loc[keys]
        has2 = "fg2fg" in w.columns
        P(f"## {'线性探针' if metric == 'linear' else 'kNN (k=20, cosine)'} 准确率\n")
        P("| 模型 | orig | fg（只剩鸟） | fg→fg（训练也涂灰） | bg_mask（鸟涂灰） | bg_box（纯背景） | orig−fg | orig−fg→fg | bg_box/orig |")
        P("|---|---|---|---|---|---|---|---|---|")
        for m in keys:
            r = w.loc[m]
            f2 = r.fg2fg if has2 else float("nan")
            P(f"| {m} | {r.orig:.3f} | {r.fg:.3f} | {f2:.3f} | {r.bg_mask:.3f} | {r.bg_box:.3f} "
              f"| {r.orig - r.fg:+.3f} | {r.orig - f2:+.3f} | {r.bg_box / r.orig:.1%} |")
        P("")
    P("## 怎么读\n")
    P("- **orig−fg**：测试时把背景拿掉掉多少。里面混着「灰背景训练时没见过」的分布偏移。")
    P("- **orig−fg→fg**：训练和测试都涂灰，排除分布偏移后的差 = 背景**信息**本身的贡献。越小越「只看鸟」。")
    P("- **bg_box / orig**：完全没有鸟（连剪影都没有）还能认出多少。远高于 0.5% 就是在靠栖息地猜物种。")
    P("- bg_mask 比 bg_box 高出的部分 = 剪影（形状）的贡献。")
    md = OUT / f"report_{kind}_C{C}.md"
    md.write_text("\n".join(L) + "\n")
    print("\n".join(L))

    import matplotlib
    matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "PingFang HK", "Heiti SC", "Arial Unicode MS"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt
    cols = ["orig", "fg", "fg2fg", "bg_mask", "bg_box"] if "fg2fg" in set(df.variant) else VARIANTS
    w = df.pivot(index="model", columns="variant", values="linear").loc[keys, cols]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(cols)); width = 0.8 / len(keys)
    colors = {"dinov2": "#888", "dinov2_reg4": "#4a90d9", "dinov3": "#d9534f"}
    for i, m in enumerate(keys):
        ax.bar(x + i * width - 0.4 + width / 2, w.loc[m].values, width, label=m, color=colors.get(m))
        for j, v in enumerate(w.loc[m].values):
            ax.text(x[j] + i * width - 0.4 + width / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.axhline(1 / 200, color="k", ls=":", lw=1, label="随机 0.5%")
    names = {"orig": "原图", "fg": "只剩鸟\n(测试涂灰)", "fg2fg": "只剩鸟\n(训练也涂灰)",
             "bg_mask": "鸟涂灰\n(剪影还在)", "bg_box": "bbox 涂灰\n(纯背景)"}
    ax.set_xticks(x); ax.set_xticklabels([names[c] for c in cols])
    ax.set_ylabel("Top-1 准确率"); ax.set_ylim(0, 1)
    ax.set_title("冻结 ViT + 线性探针：细粒度分类靠鸟还是靠背景？（CUB-200）")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / f"plot_{kind}_C{C}.png", dpi=160)
    print(f"图 → {OUT / f'plot_{kind}_C{C}.png'}")
    df.to_csv(OUT / f"acc_{kind}_C{C}.csv", index=False)


# ---------------------------------------------------------------- 入口
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "probe"])
    ap.add_argument("--models", nargs="+", default=list(vf.MODELS))
    ap.add_argument("--grid", type=int, default=16, help="patch 网格边长；16 → v2 224px / v3 256px")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--kind", default="concat", choices=["cls", "mean", "concat"])
    ap.add_argument("--C", type=float, default=1.0)
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.models, a.grid, a.bs, a.workers)
    else:
        df = probe(a.models, a.kind, a.C)
        report(df, a.kind, a.C)


if __name__ == "__main__":
    main()
