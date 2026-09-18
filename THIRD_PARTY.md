# 第三方代码与资源

| 位置 | 来源 | 协议 | 改动 |
|---|---|---|---|
| `third_party/vit_pytorch/models/` | [jeonsworld/ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch) | MIT（见同目录 LICENSE） | `modeling.py` 中 npz 的 key 改用 `"/"` 拼接（原 `os.path.join` 在 Windows 下会变反斜杠导致 KeyError） |
| DINOv3 权重 | [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)，经 timm 下载 | DINOv3 License | — |
| ImageNet-21k ViT-B/16 权重 | [google-research/vision_transformer](https://github.com/google-research/vision_transformer) | Apache-2.0 | — |
| 数据集 | CUB-200-2011 · Stanford Dogs · Stanford Cars · NABirds · IP102 | 各自的研究用途许可 | 不入库 |
