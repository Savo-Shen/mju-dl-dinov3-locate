#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_vit_tok.py —— 纯 ViT-B/16 的"只吃一部分 token"变体：冻结 DINOv3 在线打分，ViT 只看被选中的 patch。

与 train_vit.py 唯一的区别在 patch embedding 之后：按 DINOv3 的 CLS 注意力（或随机）保留 top-k 个 patch token
（CLS 永远保留），其余直接从序列里删掉——ViT 的注意力对 token 数没有要求，位置信息已加在 pos embedding 里，
不需要改模型结构。训练和测试用同样的选择。

    --select dinov3 --keep 0.25            DINOv3 选 25% token（448 输入 784 → 196）
    --select random --keep 0.25            随机选 25%（对照：选得准重不重要）
    --select none                          等价于 train_vit.py 的 raw
    --img_size 896 --keep 0.25             先放大再挑：token 数 ≈ 448 全图，主体像素 ×4
DINOv3 固定看 --sel_img（默认 448）的图，得到 28×28 注意力图后插值到 ViT 的 patch 网格，省算力。
"""
import argparse
import json
import logging
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(REPO, "weights", "imagenet21k_ViT-B_16.npz")       # ImageNet-21k 预训练权重，下载方式见 README
VIT = os.environ.get("VIT_DIR", os.path.join(REPO, "third_party", "vit_pytorch"))  # jeonsworld ViT-pytorch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_crops import Saliency  # noqa: E402   冻结 DINOv3，.maps(x) → (B,28,28) CLS 注意力
sys.path.insert(0, VIT)
from models.modeling import CONFIGS, VisionTransformer  # noqa: E402

logger = logging.getLogger(__name__)


class WarmupCosineSchedule(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total):
        self.warmup_steps, self.t_total = warmup_steps, t_total
        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return step / max(1.0, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.t_total - self.warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def loaders(args):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    rs = round(args.img_size * 600 / 448)
    tr = transforms.Compose([transforms.Resize((rs, rs), Image.BILINEAR), transforms.RandomCrop((args.img_size, args.img_size)),
                             transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
    te = transforms.Compose([transforms.Resize((rs, rs), Image.BILINEAR), transforms.CenterCrop((args.img_size, args.img_size)),
                             transforms.ToTensor(), norm])
    trs = datasets.ImageFolder(os.path.join(args.data_root, "train"), tr)
    tes = datasets.ImageFolder(os.path.join(args.data_root, "test"), te)
    assert trs.classes == tes.classes
    tl = DataLoader(trs, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers,
                    drop_last=True, pin_memory=True, persistent_workers=True)
    el = DataLoader(tes, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return tl, el, len(trs.classes)


class TokenSelectViT(nn.Module):
    """ViT-B/16 + token 选择。scorer 给每个 patch 一个分数，保留 top-k（k 固定 → 每张图算力相同）。"""

    def __init__(self, vit, grid, keep, select, scorer=None, sel_img=448):
        super().__init__()
        self.vit, self.grid, self.select, self.scorer, self.sel_img = vit, grid, select, scorer, sel_img
        self.n = grid * grid
        self.k = self.n if select == "none" else max(1, int(round(keep * self.n)))

    @torch.no_grad()
    def scores(self, x, gen=None):
        B = x.size(0)
        if self.select == "random":
            return torch.rand(B, self.n, device=x.device, generator=gen)
        xs = x if x.size(-1) == self.sel_img else F.interpolate(x, size=(self.sel_img, self.sel_img), mode="bilinear", align_corners=False)
        attn, _ = self.scorer.maps(xs)                                   # (B, 28, 28)
        a = F.avg_pool2d(attn[:, None].float(), 3, 1, 1)                 # 3×3 平滑，选出来的是连片区域而不是孤点
        if a.size(-1) != self.grid:
            a = F.interpolate(a, size=(self.grid, self.grid), mode="bilinear", align_corners=False)
        return a.flatten(1)

    def forward(self, x, gen=None):
        emb = self.vit.transformer.embeddings(x)                         # (B, 1+N, D)，pos embedding 已加
        if self.k < self.n:
            s = self.scores(x, gen)
            idx = s.topk(self.k, dim=1).indices.sort(dim=1).values + 1   # +1 跳过 CLS；排序保持原顺序（不影响结果，便于可视化）
            idx = torch.cat([torch.zeros(x.size(0), 1, dtype=idx.dtype, device=idx.device), idx], dim=1)
            emb = torch.gather(emb, 1, idx[:, :, None].expand(-1, -1, emb.size(-1)))
        enc, _ = self.vit.transformer.encoder(emb)
        return self.vit.head(enc[:, 0])


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    gen = torch.Generator(device=dev); gen.manual_seed(0)                # random 模式下测试集的选择固定
    preds, correct, n = [], 0, 0
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.cuda.amp.autocast():
            logits = model(x, gen)
        p = logits.argmax(-1); preds.append(p.cpu())
        correct += (p == y).sum().item(); n += y.numel()
    model.train()
    return correct / n, torch.cat(preds).numpy().astype(np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--output_dir", default="output")
    ap.add_argument("--pretrained", default=WEIGHTS)
    ap.add_argument("--img_size", type=int, default=448)
    ap.add_argument("--select", choices=["dinov3", "random", "none"], default="dinov3")
    ap.add_argument("--keep", type=float, default=0.25, help="保留的 patch token 比例")
    ap.add_argument("--sel_img", type=int, default=448, help="DINOv3 打分时看的分辨率（固定 448 省算力，再插值到 ViT 网格）")
    ap.add_argument("--train_batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--learning_rate", type=float, default=3e-2)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--num_steps", type=int, default=10000)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_ckpt", action="store_true", help="每次刷新 best 时把 ViT 权重存为 fp16 到 output/<name>.pt（约 170MB），供 Demo 推理")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    set_seed(args.seed)
    dev = torch.device("cuda")

    tl, el, ncls = loaders(args)
    vit = VisionTransformer(CONFIGS["ViT-B_16"], args.img_size, zero_head=True, num_classes=ncls)
    vit.load_from(np.load(args.pretrained))
    scorer = Saliency().to(dev) if args.select == "dinov3" else None
    model = TokenSelectViT(vit, args.img_size // 16, args.keep, args.select, scorer, args.sel_img).to(dev)
    logger.info("Training parameters %s", args)
    logger.info("classes=%d train=%d test=%d  tokens kept %d / %d", ncls, len(tl.dataset), len(el.dataset), model.k, model.n)

    params = [p for p in vit.parameters()]                               # 只训 ViT，DINOv3 冻结
    opt = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    sched = WarmupCosineSchedule(opt, args.warmup_steps, args.num_steps)
    scaler = torch.cuda.amp.GradScaler()
    ce = nn.CrossEntropyLoss()

    model.train()
    step, best, hist, best_preds, micro = 0, 0.0, [], None, 0
    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    while step < args.num_steps:
        for x, y in tl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.cuda.amp.autocast():
                loss = ce(model(x), y)
            scaler.scale(loss / args.grad_accum).backward()
            micro += 1
            if micro % args.grad_accum:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            step += 1
            if step % 100 == 0:
                logger.info("Training (%d / %d Steps) (loss=%.4f) lr=%.2e %.0fs", step, args.num_steps, loss.item(), sched.get_last_lr()[0], time.time() - t0)
            if step % args.eval_every == 0 or step == args.num_steps:
                acc, preds = evaluate(model, el, dev)
                if acc > best:
                    best, best_preds = acc, preds
                    if args.save_ckpt:
                        torch.save({k: v.half() for k, v in vit.state_dict().items()}, os.path.join(args.output_dir, f"{args.name}.pt"))
                hist.append((step, acc))
                logger.info("Validation Results step %d: Accuracy %.5f (best %.5f)", step, acc, best)
            if step >= args.num_steps:
                break
    logger.info("Best Accuracy: \t%f", best)
    logger.info("End Training!")
    np.savez_compressed(os.path.join(args.output_dir, f"{args.name}_preds.npz"), preds=best_preds,
                        labels=np.array([y for _, y in el.dataset.samples], dtype=np.int16),
                        paths=np.array([os.path.relpath(p, args.data_root) for p, _ in el.dataset.samples]))
    with open(os.path.join(args.output_dir, f"{args.name}.json"), "w") as fh:
        json.dump({"name": args.name, "data_root": args.data_root, "best": best, "final": hist[-1][1], "hist": hist,
                   "tokens": [model.k, model.n], "args": vars(args)}, fh, indent=1)


if __name__ == "__main__":
    main()
