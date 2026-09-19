# -*- coding: utf-8 -*-
"""三个冻结 ViT-S（dinov2 / dinov2_reg4 / dinov3）的特征对比：PCA 着色图 + NCut 前景 mask + 指标。
被 experiments/app/server.py（单独的特征 Demo）和 locate/demo/server.py（合并版 Demo）共用。"""
import base64
import io
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import vitfeat as vf  # noqa: E402

DEVICE = vf.pick_device()


def load_all(progress=None):
    """把三个模型都加载进缓存；progress(key) 用于汇报进度。"""
    for k in vf.MODELS:
        if progress: progress(k)
        vf.load_model(k, DEVICE)


def to_data_url(arr, size=320):
    """(g,g,3) float 或 (g,g) bool → 放大后的 PNG data URL，最近邻放大保持 patch 边界清晰。"""
    if arr.dtype == bool:
        arr = np.stack([arr] * 3, -1).astype(np.uint8) * 255
    else:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(arr).resize((size, size), Image.NEAREST)
    buf = io.BytesIO(); im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def img_data_url(img: Image.Image, size=320):
    im = img.convert("RGB").resize((size, size), Image.BICUBIC)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def analyze_image(img: Image.Image, keys, grid):
    out = {"original": img_data_url(img), "grid": grid, "models": []}
    for k in keys:
        t0 = time.perf_counter()
        model = vf.load_model(k, DEVICE)
        feat, size = vf.extract(model, img, grid, DEVICE)
        mask = vf.ncut_mask(feat)
        out["models"].append({
            "key": k, "name": vf.MODELS[k], "input_px": size, "patch": model.patch_embed.patch_size[0],
            "pca": to_data_url(vf.pca_rgb(feat)), "mask": to_data_url(mask),
            "fg_ratio": float(mask.mean()), "smooth": vf.smoothness(feat), "cos_med": vf.cos_median(feat),
            "ms": int((time.perf_counter() - t0) * 1000),
        })
    return out
