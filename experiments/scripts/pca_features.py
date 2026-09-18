"""把 ViT 的 patch 特征降到 3 维当 RGB 画出来，看模型有没有自己把主体分出来。

对比不同模型时按「patch 网格大小」对齐，而不是按像素——
DINOv2 是 patch14、DINOv3 是 patch16，同样 518px 出来的网格数不一样，直接比不公平。

用法：
    python experiments/scripts/pca_features.py --images experiments/data --mask
    python experiments/scripts/pca_features.py --images a.jpg --grid 40 \
        --models vit_small_patch14_dinov2.lvd142m vit_small_patch16_dinov3.lvd1689m
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# macOS 自带中文字体，否则标题里的中文会变方框
matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "PingFang HK", "Heiti SC", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def collect_images(paths):
    out = []
    for p in paths:
        p = Path(p).expanduser()
        if p.is_dir():
            out += sorted(q for q in p.iterdir() if q.suffix.lower() in IMG_EXT)
        elif p.suffix.lower() in IMG_EXT:
            out.append(p)
    if not out:
        raise SystemExit("没找到图片")
    return out


@torch.no_grad()
def patch_features(model, images, device, grid):
    """按目标 patch 网格反推输入边长，返回 (N, grid, grid, C) 特征和送进网络的图。"""
    patch = model.patch_embed.patch_size[0]
    size = grid * patch
    cfg = timm.data.resolve_model_data_config(model)

    tf = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])
    batch = torch.stack([tf(Image.open(p).convert("RGB")) for p in images]).to(device)

    feats = model.forward_features(batch)              # [B, N, C]
    feats = feats[:, model.num_prefix_tokens:, :]      # 去掉 cls / register token
    feats = feats.reshape(len(images), grid, grid, -1).float().cpu().numpy()
    return feats, batch.cpu(), cfg, size


def pca_rgb(feats):
    """所有图共用一套 PCA，颜色才可比。"""
    n, h, w, c = feats.shape
    rgb = PCA(n_components=3).fit_transform(feats.reshape(-1, c))
    lo, hi = np.percentile(rgb, 2, axis=0), np.percentile(rgb, 98, axis=0)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    return rgb.reshape(n, h, w, 3)


def otsu(x, bins=128):
    """Otsu 阈值：让前景/背景两组的类间方差最大。

    不能用中位数切——那等于强行规定一半 patch 是前景，主体小的时候必错。
    """
    hist, edges = np.histogram(x, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    csum = np.cumsum(hist * centers)
    with np.errstate(invalid="ignore", divide="ignore"):
        m0 = csum / w0
        m1 = (csum[-1] - csum) / w1
        var = w0 * w1 * (m0 - m1) ** 2
    return centers[np.nanargmax(var)]


def fg_mask(feats):
    """第一主成分二分前景/背景——DINO 那张招牌图就是这么来的。

    主成分符号是任意的，用「四条边更可能是背景」来定向。
    """
    n, h, w, c = feats.shape
    pc1 = PCA(n_components=1).fit_transform(feats.reshape(-1, c)).reshape(n, h, w)
    out = []
    for m in pc1:
        border = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]])
        if border.mean() > m.mean():          # 边缘偏高 -> 前景在低值那侧，翻过来
            m = -m
        out.append(m > otsu(m.ravel()))
    return np.stack(out)


def denorm(t, cfg):
    mean = torch.tensor(cfg["mean"]).view(3, 1, 1)
    std = torch.tensor(cfg["std"]).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=[
        "vit_small_patch14_dinov2.lvd142m",
        "vit_small_patch16_dinov3.lvd1689m",
    ])
    ap.add_argument("--grid", type=int, default=32,
                    help="目标 patch 网格边长，各模型按自己的 patch size 反推输入尺寸")
    ap.add_argument("--mask", action="store_true", help="额外画第一主成分的前景 mask")
    ap.add_argument("--out", default="experiments/outputs/pca.png")
    args = ap.parse_args()

    device = pick_device()
    images = collect_images(args.images)
    print(f"设备 {device} · 图片 {len(images)} 张 · 目标网格 {args.grid}x{args.grid}")

    rows, first = [], None
    for name in args.models:
        print(f"  {name}", flush=True)
        model = timm.create_model(
            name, pretrained=True, num_classes=0, dynamic_img_size=True
        ).eval().to(device)

        feats, shown, cfg, size = patch_features(model, images, device, args.grid)
        print(f"    输入 {size}px · patch{model.patch_embed.patch_size[0]} "
              f"· 特征维度 {feats.shape[-1]}")

        rows.append((name, size, pca_rgb(feats), fg_mask(feats) if args.mask else None))
        if first is None:
            first = (shown, cfg)
        del model, feats
        if device.type == "mps":
            torch.mps.empty_cache()

    n_col = len(images)
    n_row = 1 + len(rows) * (2 if args.mask else 1)
    fig, axes = plt.subplots(n_row, n_col,
                             figsize=(2.6 * n_col, 2.75 * n_row), squeeze=False)

    shown, cfg = first
    for j, p in enumerate(images):
        axes[0][j].imshow(denorm(shown[j], cfg))
        axes[0][j].set_title(p.stem[:20], fontsize=9)
    axes[0][0].set_ylabel("原图", fontsize=10)

    r = 1
    for name, size, rgb, mask in rows:
        short = name.split(".")[0].replace("vit_", "").replace("_patch", "/p")
        for j in range(n_col):
            axes[r][j].imshow(rgb[j])
        axes[r][0].set_ylabel(f"{short}\n{size}px · PCA", fontsize=7.5)
        r += 1
        if mask is not None:
            for j in range(n_col):
                axes[r][j].imshow(mask[j], cmap="gray")
            axes[r][0].set_ylabel(f"{short}\n前景 mask", fontsize=7.5)
            r += 1

    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"已保存 {out}")


if __name__ == "__main__":
    main()
