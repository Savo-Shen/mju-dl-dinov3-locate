#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_vit.py —— 阶段 2：纯 ViT-B/16（ImageNet-21k 预训练）在 ImageFolder 上微调。

没有选择器、没有两阶段，就是课上讲的 ViT + 一个线性分类头。同一份脚本喂 raw / dinov3 / gt 三种数据，
唯一变量是"第二阶段看到的是整图还是主体裁剪"。配方沿用 TransFG 的纯 ViT 基线：
600→RandomCrop 448 + hflip，SGD lr 3e-2 momentum 0.9，warmup 500 + cosine，10k 步，batch 16，fp16。

    python locate/train_vit.py --data_root data/crops/cub/dinov3_sq --name cub_dinov3_sq
"""
import argparse
import json
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import math

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(REPO, "weights", "imagenet21k_ViT-B_16.npz")       # ImageNet-21k 预训练权重，下载方式见 README
VIT = os.environ.get("VIT_DIR", os.path.join(REPO, "third_party", "vit_pytorch"))  # jeonsworld ViT-pytorch
sys.path.insert(0, VIT)
from models.modeling import CONFIGS, VisionTransformer  # noqa: E402


class WarmupCosineSchedule(torch.optim.lr_scheduler.LambdaLR):
    """线性 warmup 后 cosine 衰减到 0（TransFG 同款）。"""

    def __init__(self, optimizer, warmup_steps, t_total):
        self.warmup_steps, self.t_total = warmup_steps, t_total
        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return step / max(1.0, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.t_total - self.warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

logger = logging.getLogger(__name__)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def loaders(args):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    rs = round(args.img_size * 600 / 448)          # 448 → 600 的比例照搬到其他分辨率（224 → 300）
    tr = transforms.Compose([transforms.Resize((rs, rs), Image.BILINEAR), transforms.RandomCrop((args.img_size, args.img_size)),
                             transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
    te = transforms.Compose([transforms.Resize((rs, rs), Image.BILINEAR), transforms.CenterCrop((args.img_size, args.img_size)),
                             transforms.ToTensor(), norm])
    trs = datasets.ImageFolder(os.path.join(args.data_root, "train"), tr)
    tes = datasets.ImageFolder(os.path.join(args.data_root, "test"), te)
    assert trs.classes == tes.classes, "train/test 类别目录不一致"
    tl = DataLoader(trs, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers,
                    drop_last=True, pin_memory=True, persistent_workers=True)
    el = DataLoader(tes, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return tl, el, len(trs.classes)


@torch.no_grad()
def evaluate(model, loader, dev):
    """返回 (accuracy, 逐图预测 int16 数组；顺序 = ImageFolder 的 samples 顺序)。"""
    model.eval()
    preds = []
    correct = n = 0
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.cuda.amp.autocast():
            logits, _ = model(x)
        p = logits.argmax(-1)
        preds.append(p.cpu())
        correct += (p == y).sum().item()
        n += y.numel()
    model.train()
    return correct / n, torch.cat(preds).numpy().astype(np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="含 train/ test/ 的 ImageFolder 根目录")
    ap.add_argument("--name", required=True)
    ap.add_argument("--output_dir", default="output")
    ap.add_argument("--pretrained", default=WEIGHTS)
    ap.add_argument("--img_size", type=int, default=448)
    ap.add_argument("--train_batch_size", type=int, default=16, help="物理 batch；有效 batch = 物理 × grad_accum")
    ap.add_argument("--grad_accum", type=int, default=1, help="梯度累积步数（12GB 显卡用 --train_batch_size 8 --grad_accum 2）")
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
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    set_seed(args.seed)
    dev = torch.device("cuda")

    tl, el, ncls = loaders(args)
    model = VisionTransformer(CONFIGS["ViT-B_16"], args.img_size, zero_head=True, num_classes=ncls)
    model.load_from(np.load(args.pretrained))
    model.to(dev)
    logger.info("Training parameters %s", args)
    logger.info("effective batch = %d", args.train_batch_size * args.grad_accum)
    logger.info("classes=%d train=%d test=%d", ncls, len(tl.dataset), len(el.dataset))

    opt = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    sched = WarmupCosineSchedule(opt, warmup_steps=args.warmup_steps, t_total=args.num_steps)
    scaler = torch.cuda.amp.GradScaler()
    ce = nn.CrossEntropyLoss()

    model.train()
    step, best, hist, best_preds = 0, 0.0, [], None
    t0 = time.time()
    micro = 0
    opt.zero_grad(set_to_none=True)
    while step < args.num_steps:
        for x, y in tl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.cuda.amp.autocast():
                logits, _ = model(x)
                loss = ce(logits, y)
            scaler.scale(loss / args.grad_accum).backward()
            micro += 1
            if micro % args.grad_accum:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if step % 100 == 0:
                logger.info("Training (%d / %d Steps) (loss=%.4f) lr=%.2e %.0fs", step, args.num_steps, loss.item(),
                            sched.get_last_lr()[0], time.time() - t0)
            if step % args.eval_every == 0 or step == args.num_steps:
                acc, preds = evaluate(model, el, dev)
                if acc > best:
                    best, best_preds = acc, preds
                    if args.save_ckpt:
                        torch.save({k: v.half() for k, v in model.state_dict().items()}, os.path.join(args.output_dir, f"{args.name}.pt"))
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
                   "args": vars(args)}, fh, indent=1)


if __name__ == "__main__":
    main()
