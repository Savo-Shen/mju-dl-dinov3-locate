#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_crops.py —— 阶段 1：冻结 DINOv3 ViT-B/16 找主体 → bbox → 从原图裁剪，输出 ImageFolder 布局。

课堂对比实验的前处理。对每个数据集产出三个变体（都能直接喂 torchvision ImageFolder）：
    <out>/<ds>/raw/{train,test}/<cls>/<file>      原图（软链接）
    <out>/<ds>/dinov3/{train,test}/<cls>/<file>   DINOv3 自动框裁剪（无标注）
    <out>/<ds>/gt/{train,test}/<cls>/<file>       官方 bbox 裁剪（有 bbox 的数据集才做，当上界）
并写 <out>/<ds>/boxes.csv：每张图的自动框、官方框（若有）和二者 IoU。

找框：448×448 输入，取最后一层 CLS→patch 注意力（各头平均，4 个 register 吸走了 sink 伪影，见 dino_attn.py）得 28×28 图，
按 max 归一化后阈值 τ 二值化，取注意力质量最大的连通块的外接框，四边各外扩 margin，
映射回原图坐标后裁剪。可选 ncut 模式：patch 特征做 TokenCut 二分，用注意力决定哪一侧是前景。

    python locate/make_crops.py --dataset cub --calib 1000                                   # 对官方 bbox 校准 τ/margin
    python locate/make_crops.py --dataset cub --tau 0.07 --margin 0.15 --square --suffix _sq  # 最终使用的配置
"""
import argparse
import csv
import os
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dino_attn import DinoV3Attention  # noqa: E402   冻结 DINOv3 ViT-B/16 的 CLS 注意力提取（同目录）

IMG = 448
GRID = 28

# 各数据集根目录：默认取环境变量 DATA_ROOT（默认 ./data）下的子目录，也可用 --root 直接指定。目录结构见 README「数据准备」。
DATA_ROOT = os.environ.get("DATA_ROOT", "data")
ROOTS = {
    "cub": os.path.join(DATA_ROOT, "CUB_200_2011"),
    "dog": os.path.join(DATA_ROOT, "Stanford_Dogs"),
    "car": os.path.join(DATA_ROOT, "Stanford_Cars"),
    "nabirds": os.path.join(DATA_ROOT, "nabirds"),
    "ip102": os.path.join(DATA_ROOT, "IP102/classification"),
}


# ----------------------------------------------------------------------------- 数据集列表
def list_cub(root):
    ids = {}
    for line in open(os.path.join(root, "images.txt")):
        i, p = line.split()
        ids[i] = {"path": os.path.join(root, "images", p), "cls": p.split("/")[0], "file": os.path.basename(p)}
    for line in open(os.path.join(root, "train_test_split.txt")):
        i, s = line.split()
        ids[i]["split"] = "train" if s == "1" else "test"
    for line in open(os.path.join(root, "bounding_boxes.txt")):
        i, x, y, w, h = line.split()
        x, y, w, h = map(float, (x, y, w, h))
        ids[i]["gt"] = (x, y, x + w, y + h)
    return list(ids.values())


def list_nabirds(root):
    ids = {}
    for line in open(os.path.join(root, "images.txt")):
        i, p = line.split()
        ids[i] = {"path": os.path.join(root, "images", p), "cls": p.split("/")[0], "file": os.path.basename(p)}
    for line in open(os.path.join(root, "train_test_split.txt")):
        i, s = line.split()
        ids[i]["split"] = "train" if s == "1" else "test"
    for line in open(os.path.join(root, "bounding_boxes.txt")):
        i, x, y, w, h = line.split()
        x, y, w, h = map(float, (x, y, w, h))
        ids[i]["gt"] = (x, y, x + w, y + h)
    return list(ids.values())


def list_dog(root):
    from scipy import io
    items = []
    for split, mat in (("train", "train_list.mat"), ("test", "test_list.mat")):
        m = io.loadmat(os.path.join(root, mat))
        for f in m["file_list"][:, 0]:
            rel = str(f[0])
            ann = os.path.join(root, "Annotation", rel[:-4])
            gt = None
            if os.path.exists(ann):
                bb = ET.parse(ann).getroot().find("object").find("bndbox")
                gt = tuple(float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
            items.append({"path": os.path.join(root, "Images", rel), "cls": rel.split("/")[0],
                          "file": os.path.basename(rel), "split": split, "gt": gt})
    return items


def list_car(root):
    items = []
    for split in ("train", "test"):
        anno = {}
        for row in csv.reader(open(os.path.join(root, f"anno_{split}.csv"))):
            anno[row[0]] = tuple(map(float, row[1:5]))
        d = os.path.join(root, split)
        for cls in sorted(os.listdir(d)):
            for f in sorted(os.listdir(os.path.join(d, cls))):
                items.append({"path": os.path.join(d, cls, f), "cls": cls, "file": f, "split": split,
                              "gt": anno.get(f)})
    return items


def list_ip102(root):
    items = []
    for split in ("train", "test"):
        d = os.path.join(root, split)
        for cls in sorted(os.listdir(d)):
            for f in sorted(os.listdir(os.path.join(d, cls))):
                items.append({"path": os.path.join(d, cls, f), "cls": cls, "file": f, "split": split, "gt": None})
    return items


LISTERS = {"cub": list_cub, "dog": list_dog, "car": list_car, "nabirds": list_nabirds, "ip102": list_ip102}


class ImgDS(Dataset):
    def __init__(self, items):
        self.items = items
        self.tf = transforms.Compose([
            transforms.Resize((IMG, IMG), Image.BILINEAR),      # IMG 由 --img 决定
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        im = Image.open(self.items[i]["path"]).convert("RGB")
        w, h = im.size
        return self.tf(im), torch.tensor([w, h], dtype=torch.float32), i


# ----------------------------------------------------------------------------- DINOv3 显著性
Saliency = DinoV3Attention        # .maps(x) → (B, 28, 28) CLS 注意力 + (B, 784, C) patch 特征；train_vit_tok.py 也用它


@torch.no_grad()
def ncut_fg(feats, attn, sim_tau=0.2):
    """TokenCut 式二分：cos 相似度图 → 归一化拉普拉斯第二特征向量 → 中位数二分；注意力大的一侧当前景。"""
    f = F.normalize(feats.float(), dim=-1)
    W = f @ f.transpose(1, 2)
    W = torch.where(W >= sim_tau, torch.ones_like(W), torch.full_like(W, 1e-5))
    d = W.sum(-1)
    dis = d.rsqrt()
    L = torch.eye(W.size(1), device=W.device)[None] - dis[:, :, None] * W * dis[:, None, :]
    evals, evecs = torch.linalg.eigh(L)
    v = evecs[:, :, 1] * dis
    fg = v > v.median(dim=1, keepdim=True).values
    a = attn.flatten(1)
    fg_attn = (a * fg).sum(1) / fg.sum(1).clamp(min=1)
    bg_attn = (a * ~fg).sum(1) / (~fg).sum(1).clamp(min=1)
    flip = bg_attn > fg_attn
    fg[flip] = ~fg[flip]
    return fg.reshape(-1, GRID, GRID)


def adjust_box(box, margin, min_frac, square):
    """448 空间的框 → 外扩 margin → 最小边约束 → （可选）补成正方形（保持与整图 pipeline 相同的长宽比，避免
    Resize((600,600)) 把细长框拉变形）→ 裁到图内。"""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, x2 = x1 - margin * w, x2 + margin * w
    y1, y2 = y1 - margin * h, y2 + margin * h
    mn = min_frac * IMG
    if x2 - x1 < mn:
        c = (x1 + x2) / 2
        x1, x2 = c - mn / 2, c + mn / 2
    if y2 - y1 < mn:
        c = (y1 + y2) / 2
        y1, y2 = c - mn / 2, c + mn / 2
    if square:
        side = min(max(x2 - x1, y2 - y1), IMG)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        x1, x2, y1, y2 = cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2
        if x1 < 0: x1, x2 = 0.0, side
        if y1 < 0: y1, y2 = 0.0, side
        if x2 > IMG: x1, x2 = IMG - side, float(IMG)
        if y2 > IMG: y1, y2 = IMG - side, float(IMG)
    return (max(0.0, x1), max(0.0, y1), min(float(IMG), x2), min(float(IMG), y2))


def mask_to_box(mask, weight, margin, min_frac, square=False):
    """二值 28×28 → 取权重和最大的连通块 → 448 像素框 → adjust_box。返回 (x1,y1,x2,y2) in 448 空间。"""
    lab, n = ndimage.label(mask)
    if n == 0:
        return (0.0, 0.0, float(IMG), float(IMG))
    best = max(range(1, n + 1), key=lambda k: weight[lab == k].sum())
    ys, xs = np.where(lab == best)
    cell = IMG / GRID
    box = (xs.min() * cell, ys.min() * cell, (xs.max() + 1) * cell, (ys.max() + 1) * cell)
    return adjust_box(box, margin, min_frac, square)


def gt_adjusted(gt, w, h, margin, min_frac, square):
    """官方框映射到 448 空间，做与自动框相同的 margin/square 处理，再映回原图。"""
    sx, sy = IMG / w, IMG / h
    b = adjust_box((gt[0] * sx, gt[1] * sy, gt[2] * sx, gt[3] * sy), margin, min_frac, square)
    return to_orig(b, w, h)


def to_orig(box, w, h):
    sx, sy = w / IMG, h / IMG
    return (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def boxes_from_maps(attn, feats, method, tau, margin, min_frac, square=False):
    """attn (B,28,28) feats (B,784,C) → list of 448 空间框。"""
    a = attn.float()
    a = F.avg_pool2d(a[:, None], 3, 1, 1)[:, 0]                       # 3×3 平滑
    a = a / a.flatten(1).max(1).values[:, None, None].clamp(min=1e-8)
    if method == "attn":
        masks = (a >= tau)
    else:
        masks = ncut_fg(feats, a)
    masks = masks.cpu().numpy()
    a = a.cpu().numpy()
    return [mask_to_box(masks[i], a[i], margin, min_frac, square) for i in range(len(masks))]


def crop_save(src, box, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    im = Image.open(src).convert("RGB")
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x2, y2 = max(x2, x1 + 8), max(y2, y1 + 8)
    im.crop((x1, y1, x2, y2)).save(dst, quality=95)


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        return
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:                       # Windows 非管理员建不了软链接：退化为复制
        shutil.copyfile(src, dst)


# ----------------------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(ROOTS))
    ap.add_argument("--root", default=None, help="数据集根目录（默认 $DATA_ROOT/<数据集目录名>）")
    ap.add_argument("--out", default=os.environ.get("CROPS_DIR", "data/crops"), help="裁剪输出根目录")
    ap.add_argument("--method", choices=["attn", "ncut"], default="attn")
    ap.add_argument("--tau", type=float, default=0.2, help="attn 模式：max 归一化后的二值化阈值")
    ap.add_argument("--margin", type=float, default=0.1, help="框四边各外扩的比例（相对框宽/高）")
    ap.add_argument("--min_frac", type=float, default=0.2, help="框最短边不小于图边的比例")
    ap.add_argument("--square", action="store_true", help="框补成正方形（自动框和官方框都做），避免细长框被拉变形")
    ap.add_argument("--suffix", default="", help="变体目录后缀，如 _sq → dinov3_sq / gt_sq；raw 不受影响")
    ap.add_argument("--img", type=int, default=448, help="DINOv3 输入边长（224 时找框只花 1/4 算力，网格 14×14）")
    ap.add_argument("--calib", type=int, default=0, help=">0：只在训练集前 N 张有 bbox 的图上扫 τ/margin，不产出裁剪")
    ap.add_argument("--taus", default="0.1,0.15,0.2,0.3,0.4", help="calib 扫描的 τ 列表")
    ap.add_argument("--margins", default="0,0.1,0.2", help="calib 扫描的 margin 列表")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    global IMG, GRID
    IMG, GRID = args.img, args.img // 16

    items = LISTERS[args.dataset](args.root or ROOTS[args.dataset])
    has_gt = any(it["gt"] is not None for it in items)
    print(f"[{args.dataset}] {len(items)} images, gt bbox: {has_gt}", flush=True)

    if args.calib:
        items = [it for it in items if it["split"] == "train" and it["gt"] is not None][: args.calib]

    dev = torch.device("cuda")
    sal = Saliency().to(dev)
    loader = DataLoader(ImgDS(items), batch_size=args.batch, num_workers=args.workers, pin_memory=True)

    if args.calib:
        taus = [float(t) for t in args.taus.split(",")]
        margins = [float(m) for m in args.margins.split(",")]
        grid = [("attn", t, m) for t in taus for m in margins] + [("ncut", 0.0, m) for m in margins]
        ious = {g: [] for g in grid}
        for x, wh, idx in loader:
            attn, feats = sal.maps(x.to(dev, non_blocking=True))
            for g in grid:
                bx = boxes_from_maps(attn, feats, g[0], g[1], g[2], args.min_frac)
                for b, (w, h), i in zip(bx, wh.tolist(), idx.tolist()):
                    ious[g].append(iou(to_orig(b, w, h), items[i]["gt"]))
        print(f"\n校准（{len(items)} 张，min_frac={args.min_frac}）：method  tau  margin  meanIoU  IoU>=0.5")
        for g in grid:
            v = np.array(ious[g])
            print(f"  {g[0]:5s} {g[1]:4.2f} {g[2]:4.1f}   {v.mean():.3f}   {(v >= 0.5).mean():.3f}")
        return

    out = os.path.join(args.out, args.dataset)
    os.makedirs(out, exist_ok=True)
    pool = ThreadPoolExecutor(args.workers)
    futs = []
    rows = []
    t0 = time.time()
    for bi, (x, wh, idx) in enumerate(loader):
        attn, feats = sal.maps(x.to(dev, non_blocking=True))
        bx = boxes_from_maps(attn, feats, args.method, args.tau, args.margin, args.min_frac, args.square)
        for b, (w, h), i in zip(bx, wh.tolist(), idx.tolist()):
            it = items[i]
            box = to_orig(b, w, h)
            rel = os.path.join(it["split"], it["cls"], it["file"])
            link(it["path"], os.path.join(out, "raw", rel))
            futs.append(pool.submit(crop_save, it["path"], box, os.path.join(out, "dinov3" + args.suffix, rel)))
            gi = ""
            if it["gt"] is not None:
                gbox = gt_adjusted(it["gt"], w, h, args.margin, args.min_frac, args.square)
                futs.append(pool.submit(crop_save, it["path"], gbox, os.path.join(out, "gt" + args.suffix, rel)))
                gi = f"{iou(box, it['gt']):.4f}"          # IoU 仍按原始官方框算，便于跨变体比较
            rows.append([it["path"], it["split"], it["cls"], *[f"{v:.1f}" for v in box],
                         *([f"{v:.1f}" for v in it["gt"]] if it["gt"] else ["", "", "", ""]), gi, f"{w:.0f}", f"{h:.0f}"])
        if bi % 50 == 0:
            print(f"  {len(rows)}/{len(items)}  {time.time() - t0:.0f}s", flush=True)
    for f in futs:
        f.result()
    with open(os.path.join(out, f"boxes{args.suffix}.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["path", "split", "cls", "x1", "y1", "x2", "y2", "gx1", "gy1", "gx2", "gy2", "iou", "W", "H"])
        wr.writerows(rows)
    if has_gt:
        v = np.array([float(r[11]) for r in rows if r[11]])
        for split in ("train", "test"):
            vs = np.array([float(r[11]) for r in rows if r[11] and r[1] == split])
            print(f"  IoU(auto, gt) {split}: mean {vs.mean():.3f}  >=0.5 {(vs >= 0.5).mean():.3f}  n={len(vs)}")
        print(f"  IoU(auto, gt) all:   mean {v.mean():.3f}  >=0.5 {(v >= 0.5).mean():.3f}")
    print(f"[{args.dataset}] done {len(rows)} images in {time.time() - t0:.0f}s → {out}", flush=True)


if __name__ == "__main__":
    main()
