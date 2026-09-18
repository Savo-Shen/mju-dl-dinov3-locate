# experiments/app/ — Web Demo

```bash
conda activate dinov3 && export HF_ENDPOINT=https://hf-mirror.com
bash experiments/app/run.sh        # http://127.0.0.1:8000
```

`server.py`（FastAPI + uvicorn）加载三个模型：`dinov2`（ViT-S/14，448 输入）、`dinov2_reg4`（加 4 个 register）、`dinov3`（ViT-S/16，512 输入）。上传一张图，返回三者并排的 PCA 着色图（前 3 个主成分当 RGB，同色 = 特征相近）和自动前景 mask；有 CUB 官方分割的图会顺带算 IoU。特征提取在 `../scripts/vitfeat.py`。首次运行下载三份权重（各约 90MB），之后 M1 Max 上单图约 1.5 秒。
