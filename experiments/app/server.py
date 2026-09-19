"""上传一张图 → 三个冻结 ViT 并排输出 PCA 特征图、前景 mask 和指标。

    conda activate dinov3
    HF_ENDPOINT=https://hf-mirror.com python experiments/app/server.py
    open http://127.0.0.1:8000
"""

import sys
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import features as ft  # noqa: E402
import vitfeat as vf  # noqa: E402

app = FastAPI(title="DINOv3 特征可视化")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

SAMPLES = HERE.parent / "data"
DEVICE = ft.DEVICE
_infer_lock = threading.Lock()   # MPS 不适合并发推理，串行化
_status = {"ready": False, "loading": None, "error": None}


def _warmup():
    try:
        ft.load_all(lambda k: _status.__setitem__("loading", k))
        _status["ready"] = True
        _status["loading"] = None
    except Exception as e:  # noqa: BLE001
        _status["error"] = str(e)


threading.Thread(target=_warmup, daemon=True).start()


def analyze_image(img, keys, grid):
    with _infer_lock:
        return ft.analyze_image(img, keys, grid)


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
