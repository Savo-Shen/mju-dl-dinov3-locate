"""公共模块：加载冻结 ViT、抽 patch 特征、算可视化与指标。

pca_features.py / bg_quant.py / app 都从这里取函数，避免三处各写一份。
"""

import threading

import numpy as np
import timm
import torch
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms

MODELS = {
    "dinov2":      "vit_small_patch14_dinov2.lvd142m",
    "dinov2_reg4": "vit_small_patch14_reg4_dinov2.lvd142m",
    "dinov3":      "vit_small_patch16_dinov3.lvd1689m",
}

_cache: dict[str, torch.nn.Module] = {}
_lock = threading.Lock()


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(key, device=None):
    """按 key 取模型，只加载一次。"""
    device = device or pick_device()
    with _lock:
        if key not in _cache:
            m = timm.create_model(MODELS[key], pretrained=True, num_classes=0,
                                  dynamic_img_size=True).eval().to(device)
            _cache[key] = m
        return _cache[key]


def loaded_keys():
    return list(_cache)


@torch.no_grad()
def extract(model, img: Image.Image, grid: int, device=None):
    """单张图 → (grid, grid, C) 的 patch 特征。输入边长按 grid × patch_size 反推。"""
    device = device or pick_device()
    patch = model.patch_embed.patch_size[0]
    size = grid * patch
    cfg = timm.data.resolve_model_data_config(model)
    tf = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])
    x = tf(img.convert("RGB")).unsqueeze(0).to(device)
    f = model.forward_features(x)[:, model.num_prefix_tokens:, :]
    return f.reshape(grid, grid, -1).float().cpu().numpy(), size


# ---------------------------------------------------------------- 可视化
def pca_rgb(feat):
    """(g, g, C) → (g, g, 3)，前三主成分当 RGB，按 2/98 分位数拉伸。"""
    g, _, c = feat.shape
    rgb = PCA(n_components=3).fit_transform(feat.reshape(-1, c))
    lo, hi = np.percentile(rgb, 2, axis=0), np.percentile(rgb, 98, axis=0)
    return np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1).reshape(g, g, 3)


def otsu(x, bins=128):
    """Otsu 阈值：让前景/背景两组的类间方差最大。"""
    hist, edges = np.histogram(x, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    csum = np.cumsum(hist * centers)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = w0 * w1 * (csum / w0 - (csum[-1] - csum) / w1) ** 2
    return centers[np.nanargmax(var)]


def pc1_mask(feat):
    """第一主成分二分前景/背景。符号任意，用「四条边更可能是背景」定向。"""
    g = feat.shape[0]
    pc1 = PCA(n_components=1).fit_transform(feat.reshape(-1, feat.shape[-1])).reshape(g, g)
    border = np.concatenate([pc1[0], pc1[-1], pc1[:, 0], pc1[:, -1]])
    if border.mean() > pc1.mean():
        pc1 = -pc1
    return pc1 > otsu(pc1.ravel())


def ncut_mask(feat, tau=None, eps=1e-5):
    """TokenCut 风格的前景分割：patch 余弦相似度图上做 Normalized Cut。

    W_ij = 1 if cos(f_i, f_j) > tau else eps；解 (D - W) v = λ D v 的第二小特征向量
    （Fiedler 向量），按 0 二分。取「和四条边重叠更少」的那一侧当前景。

    tau 默认取**该图相似度矩阵的中位数**而不是 TokenCut 原文的 0.2：
    不同模型的相似度尺度差三倍（DINOv2 中位数 ~0.2、DINOv3 ~0.6），固定阈值会让
    DINOv3 的图几乎全连通、切不开；按中位数则各模型图密度都是 50%，可比。
    比单图 PC1 稳：PC1 抓的是方差最大的方向，不一定是主体 vs 背景。
    """
    from scipy.linalg import eigh
    g = feat.shape[0]
    f = feat.reshape(-1, feat.shape[-1])
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
    C = f @ f.T
    thr = np.median(C) if tau is None else tau
    W = np.where(C > thr, 1.0, eps)
    D = np.diag(W.sum(1))
    _, vecs = eigh(D - W, D, subset_by_index=[0, 1])   # 只要最小的两个
    m = vecs[:, 1].reshape(g, g) > 0
    border = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]])
    if border.mean() > 0.5:       # 前景那侧不该占满边框
        m = ~m
    return m


# ---------------------------------------------------------------- 指标
def smoothness(feat):
    """patch 与 4 邻域的平均余弦相似度，越高越平滑。"""
    f = feat / (np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-8)
    return float(np.mean([
        (f[1:] * f[:-1]).sum(-1).mean(),
        (f[:, 1:] * f[:, :-1]).sum(-1).mean(),
    ]))


def artifact_ratio(feat):
    """范数离群 patch 的比例（median + 3·MAD 之外），衡量 artifact token。"""
    norms = np.linalg.norm(feat, axis=-1).ravel()
    med = np.median(norms)
    mad = np.median(np.abs(norms - med)) + 1e-8
    return float(((norms - med) > 3 * 1.4826 * mad).mean())


def cos_median(feat):
    """patch 两两余弦相似度的中位数——衡量特征「有多像」。
    DINOv2 约 0.2、DINOv3 约 0.6，差三倍；这就是固定阈值 NCut 在 v3 上切不开的原因。"""
    f = feat.reshape(-1, feat.shape[-1])
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
    return float(np.median(f @ f.T))


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else float("nan")
