# -*- coding: utf-8 -*-
"""
Demo（合并版）：一张图，两个页签。
  ① 找主体 → 分类：DINOv3 CLS 注意力找框 → 三个 ViT-B/16（原图 / 裁剪 / 25% token）并排给出预测
  ② 三个冻结 ViT 在看什么：dinov2 / dinov2_reg4 / dinov3 的 patch 特征 PCA 着色图 + NCut 前景 mask（原 experiments/app）

    原图分类器        看整张图                         （方向一的基线）
    裁剪分类器        看 DINOv3 框出并放大的主体       （方向一：先找再放大）
    25% token 分类器  只看 DINOv3 挑出的 1/4 个 patch   （方向二：换算力）

三个分类器都是 NABirds（555 类）上微调的 ViT-B/16，权重放在 weights/：
    nabirds_raw_ckpt.pt  nabirds_dinov3_sq_ckpt.pt  nabirds_tok25_ckpt.pt
DINOv3 ViT-B/16 由 timm 自动下载。Mac 上用 MPS，整条链路每张图约 0.1～0.2 秒。

    conda activate dinov3 && export HF_ENDPOINT=https://hf-mirror.com
    python locate/demo/server.py            # → http://127.0.0.1:8001
"""
import base64
import io
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
from torchvision import transforms

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "locate"))
sys.path.insert(0, str(REPO / "third_party" / "vit_pytorch"))
sys.path.insert(0, str(REPO / "experiments" / "app"))
import features as ft  # noqa: E402   三个冻结 ViT-S 的特征对比（PCA / NCut）
import vitfeat as vf  # noqa: E402
from dino_attn import DinoV3Attention  # noqa: E402
from make_crops import IMG, adjust_box, mask_to_box, to_orig  # noqa: E402
from models.modeling import CONFIGS, VisionTransformer  # noqa: E402

WEIGHTS = REPO / "weights"
SAMPLE_DIRS = [REPO / "data" / "demo_samples", REPO / "experiments" / "data"]   # 两个 Demo 的示例图都列出来
CLASSES = json.load(open(HERE / "nabirds_classes.json"))
TAU, MARGIN, MIN_FRAC, KEEP = 0.07, 0.15, 0.2, 0.25          # 与训练时 make_crops / train_vit_tok 完全一致
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="DINOv3 找主体 → ViT 分类")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
_lock = threading.Lock()
_status = {"ready": False, "loading": None, "error": None}
M = {}

NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TF_CLS = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR), transforms.CenterCrop((448, 448)),
                             transforms.ToTensor(), NORM])                       # 分类器的测试变换（与训练一致）
TF_BOX = transforms.Compose([transforms.Resize((IMG, IMG), Image.BILINEAR), transforms.ToTensor(), NORM])  # 找框


def _load_vit(name):
    vit = VisionTransformer(CONFIGS["ViT-B_16"], 448, zero_head=True, num_classes=len(CLASSES))
    sd = torch.load(WEIGHTS / f"{name}.pt", map_location="cpu")
    vit.load_state_dict({k: v.float() for k, v in sd.items()})
    return vit.to(DEVICE).eval()


def _warmup():
    try:
        _status["loading"] = "DINOv3 ViT-B/16"
        M["dino"] = DinoV3Attention().to(DEVICE)
        for key, name in (("raw", "nabirds_raw_ckpt"), ("crop", "nabirds_dinov3_sq_ckpt"), ("tok", "nabirds_tok25_ckpt")):
            _status["loading"] = name
            if (WEIGHTS / f"{name}.pt").exists():
                M[key] = _load_vit(name)
            else:                                                            # 缺哪个权重就少哪一列，页面照常能用
                M[key] = None; _status.setdefault("missing", []).append(name)
        if M["raw"] is None:
            raise FileNotFoundError("weights/nabirds_raw_ckpt.pt 不存在，见 weights/README.md")
        with torch.no_grad():                                                    # 预热一次，避免第一张图慢
            x = torch.zeros(1, 3, 448, 448, device=DEVICE)
            M["dino"].maps(x); M["raw"](x)
        ft.load_all(lambda k: _status.__setitem__("loading", f"特征对比 · {k}"))  # 三个 ViT-S
        _status["ready"], _status["loading"] = True, None
    except Exception as e:  # noqa: BLE001
        _status["error"] = f"{type(e).__name__}: {e}"


threading.Thread(target=_warmup, daemon=True).start()


# ---------------------------------------------------------------- 图像工具
def data_url(im: Image.Image, fmt="JPEG"):
    buf = io.BytesIO(); im.save(buf, fmt, quality=90) if fmt == "JPEG" else im.save(buf, fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def heat_overlay(img: Image.Image, attn: np.ndarray, size=448):
    """注意力图 (28,28) → 归一化 → 双线性放大 → 红黄伪彩，叠在缩放后的原图上。"""
    a = attn / max(attn.max(), 1e-8)
    a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)) / 255.0
    base = np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR)).astype(np.float32) / 255.0
    color = np.stack([np.clip(a * 2, 0, 1), np.clip(a * 2 - 0.6, 0, 1), np.zeros_like(a)], -1)   # 黑→红→黄
    out = base * (1 - 0.55 * a[..., None]) + color * (0.55 * a[..., None]) * 1.6
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


def box_overlay(img: Image.Image, box, size=448):
    im = img.convert("RGB").copy(); w, h = im.size
    d = ImageDraw.Draw(im); d.rectangle(box, outline=(255, 40, 40), width=max(3, w // 150))
    return im.resize((size, int(size * h / w)) if w >= h else (int(size * w / h), size), Image.BILINEAR)


def token_overlay(img448: Image.Image, keep_mask: np.ndarray, size=448):
    """被丢掉的 patch 压暗，保留的 patch 原样：直观看到 ViT 实际看了哪 25%。"""
    base = np.asarray(img448.convert("RGB").resize((size, size), Image.BILINEAR)).astype(np.float32)
    m = np.asarray(Image.fromarray((keep_mask * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)) / 255.0
    out = base * (0.18 + 0.82 * m[..., None])
    return Image.fromarray(out.astype(np.uint8))


def topk(logits, k=5):
    p = logits.softmax(-1)[0]
    v, i = p.topk(k)
    return [{"name": CLASSES[j], "p": float(pv)} for pv, j in zip(v.tolist(), i.tolist())]


def _sync():
    """MPS / CUDA 是异步的，计时前同步一下，否则 ms 只是发射时间。"""
    if DEVICE.type == "mps": torch.mps.synchronize()
    elif DEVICE.type == "cuda": torch.cuda.synchronize()


# ---------------------------------------------------------------- 推理
@torch.no_grad()
def run(img: Image.Image):
    img = img.convert("RGB"); W, H = img.size
    t = {}
    # 1) DINOv3 找主体（与 make_crops.py 相同：整图缩到 448，最后一层 CLS 注意力）
    t0 = time.perf_counter()
    attn, _ = M["dino"].maps(TF_BOX(img)[None].to(DEVICE))
    a = F.avg_pool2d(attn[:, None].float(), 3, 1, 1)[:, 0]
    a = a / a.flatten(1).max(1).values[:, None, None].clamp(min=1e-8)
    a_np = a[0].cpu().numpy()
    box448 = mask_to_box((a_np >= TAU), a_np, MARGIN, MIN_FRAC, square=True)
    box = to_orig(box448, W, H)
    _sync(); t["find"] = time.perf_counter() - t0
    crop = img.crop(tuple(int(round(v)) for v in box))

    # 2) 原图分类
    t0 = time.perf_counter(); x_raw = TF_CLS(img)[None].to(DEVICE)
    logits_raw, _ = M["raw"](x_raw); _sync(); t["raw"] = time.perf_counter() - t0
    preds = {"raw": topk(logits_raw)}
    # 3) 裁剪分类
    if M["crop"] is not None:
        t0 = time.perf_counter(); x_crop = TF_CLS(crop)[None].to(DEVICE)
        logits_crop, _ = M["crop"](x_crop); _sync(); t["crop"] = time.perf_counter() - t0
        preds["crop"] = topk(logits_crop)
    # 4) 只看 25% token（与 train_vit_tok.py 一致：在分类器输入图上重新打分，3×3 平滑，top-k）
    t0 = time.perf_counter()
    attn2, _ = M["dino"].maps(x_raw)
    s = F.avg_pool2d(attn2[:, None].float(), 3, 1, 1).flatten(1)
    k = int(round(KEEP * s.size(1)))
    idx = s.topk(k, dim=1).indices.sort(dim=1).values + 1
    idx = torch.cat([torch.zeros(1, 1, dtype=idx.dtype, device=idx.device), idx], 1)
    if M["tok"] is not None:
        emb = M["tok"].transformer.embeddings(x_raw)
        emb = torch.gather(emb, 1, idx[:, :, None].expand(-1, -1, emb.size(-1)))
        enc, _ = M["tok"].transformer.encoder(emb)
        preds["tok"] = topk(M["tok"].head(enc[:, 0]))
    _sync(); t["tok"] = time.perf_counter() - t0
    keep = np.zeros(s.size(1), dtype=np.float32); keep[(idx[0, 1:] - 1).cpu().numpy()] = 1
    keep = keep.reshape(28, 28)

    img448 = transforms.CenterCrop(448)(transforms.Resize((600, 600), Image.BILINEAR)(img))
    return {
        "size": [W, H], "box": [round(v) for v in box],
        "box_frac": round((box[2] - box[0]) * (box[3] - box[1]) / (W * H), 3),
        "images": {
            "boxed": data_url(box_overlay(img, box)),
            "heat": data_url(heat_overlay(img, a_np)),
            "crop": data_url(crop.resize((448, 448), Image.BILINEAR)),
            "tokens": data_url(token_overlay(img448, keep)),
        },
        "preds": preds,
        "ms": {k_: int(v * 1000) for k_, v in t.items()},
        "tokens": [k, s.size(1)],
    }


# ---------------------------------------------------------------- 路由
@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/status")
def status():
    return {**_status, "device": str(DEVICE), "classes": len(CLASSES)}


@app.get("/api/samples")
def samples():
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    out = []
    for i, d in enumerate(SAMPLE_DIRS):
        if d.exists():
            out += [{"name": p.stem, "url": f"/api/sample/{i}/{p.name}", "group": d.name}
                    for p in sorted(d.iterdir()) if p.suffix.lower() in exts]
    return out


@app.get("/api/sample/{i}/{name}")
def sample(i: int, name: str):
    d = SAMPLE_DIRS[i] if 0 <= i < len(SAMPLE_DIRS) else None
    p = d / name if d else None
    if p is None or not p.is_file() or p.parent != d:
        raise HTTPException(404)
    return FileResponse(p)


@app.post("/api/analyze")
def analyze(file: UploadFile = File(...), models: str = Form("dinov2,dinov2_reg4,dinov3"), grid: int = Form(32)):
    """页签②：三个冻结 ViT-S 的 PCA 着色图 + NCut mask。"""
    if _status["error"]:
        raise HTTPException(500, _status["error"])
    if not _status["ready"]:
        raise HTTPException(503, f"模型加载中：{_status['loading']}")
    keys = [k for k in models.split(",") if k in vf.MODELS]
    if not keys:
        raise HTTPException(400, "没有选中任何模型")
    try:
        img = Image.open(io.BytesIO(file.file.read()))
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "不是能打开的图片")
    with _lock:
        return JSONResponse(ft.analyze_image(img, keys, max(8, min(int(grid), 48))))


@app.post("/api/classify")
def classify(file: UploadFile = File(...)):
    if _status["error"]:
        raise HTTPException(500, _status["error"])
    if not _status["ready"]:
        raise HTTPException(503, f"模型加载中：{_status['loading']}")
    try:
        img = Image.open(io.BytesIO(file.file.read()))
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "不是能打开的图片")
    with _lock:
        return JSONResponse(run(img))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")
