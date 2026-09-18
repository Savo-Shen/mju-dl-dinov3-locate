"""上传一张图 → 三个冻结 ViT 并排输出 PCA 特征图、前景 mask 和指标。

    conda activate dinov3
    HF_ENDPOINT=https://hf-mirror.com python experiments/app/server.py
    open http://127.0.0.1:8000
"""

import base64
import io
import sys
import threading
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import vitfeat as vf  # noqa: E402

app = FastAPI(title="DINOv3 特征可视化")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

SAMPLES = HERE.parent / "data"
DEVICE = vf.pick_device()
_infer_lock = threading.Lock()   # MPS 不适合并发推理，串行化
_status = {"ready": False, "loading": None, "error": None}


def _warmup():
    try:
        for k in vf.MODELS:
            _status["loading"] = k
            vf.load_model(k, DEVICE)
        _status["ready"] = True
        _status["loading"] = None
    except Exception as e:  # noqa: BLE001
        _status["error"] = str(e)


threading.Thread(target=_warmup, daemon=True).start()


# ---------------------------------------------------------------- 工具
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
    with _infer_lock:
        for k in keys:
            t0 = time.perf_counter()
            model = vf.load_model(k, DEVICE)
            feat, size = vf.extract(model, img, grid, DEVICE)
            mask = vf.ncut_mask(feat)
            out["models"].append({
                "key": k,
                "name": vf.MODELS[k],
                "input_px": size,
                "patch": model.patch_embed.patch_size[0],
                "pca": to_data_url(vf.pca_rgb(feat)),
                "mask": to_data_url(mask),
                "fg_ratio": float(mask.mean()),
                "smooth": vf.smoothness(feat),
                "cos_med": vf.cos_median(feat),
                "ms": int((time.perf_counter() - t0) * 1000),
            })
    return out


# ---------------------------------------------------------------- 路由
@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/status")
def status():
    return {**_status, "device": str(DEVICE), "models": list(vf.MODELS)}


@app.get("/api/samples")
def samples():
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = sorted(p for p in SAMPLES.iterdir() if p.suffix.lower() in exts) if SAMPLES.exists() else []
    return [{"name": p.stem, "url": f"/api/sample/{p.name}"} for p in files]


@app.get("/api/sample/{name}")
def sample(name: str):
    p = SAMPLES / name
    if not p.is_file() or p.parent != SAMPLES:
        raise HTTPException(404)
    return FileResponse(p)


@app.post("/api/analyze")
def analyze(
    file: UploadFile = File(...),
    models: str = Form("dinov2,dinov2_reg4,dinov3"),
    grid: int = Form(32),
):
    if not _status["ready"]:
        raise HTTPException(503, f"模型加载中：{_status['loading']}")
    keys = [k for k in models.split(",") if k in vf.MODELS]
    if not keys:
        raise HTTPException(400, "没有选中任何模型")
    grid = max(8, min(int(grid), 48))
    try:
        img = Image.open(io.BytesIO(file.file.read()))
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "不是能打开的图片")
    return JSONResponse(analyze_image(img, keys, grid))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
