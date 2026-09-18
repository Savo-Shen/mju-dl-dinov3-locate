# third_party/

`vit_pytorch/models/`：[jeonsworld/ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch) 的 ViT 模型定义（MIT，见 `vit_pytorch/LICENSE`），用来加载 Google 的 ImageNet-21k `.npz` 权重并在 448 输入下插值位置编码。唯一改动：`modeling.py` 里 npz 的 key 用 `"/"` 拼接而不是 `os.path.join`，否则 Windows 下会变成反斜杠导致 KeyError。`train_vit.py` / `train_vit_tok.py` 默认从这里 import；设 `VIT_DIR` 可指向别处。
