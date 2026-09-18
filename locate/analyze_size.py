#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按主体大小分组比较各输入变体的测试准确率。
用法: python locate/analyze_size.py <ds> <boxes.csv 或 .csv.gz> <output_dir> name1 name2 ...
例:   python locate/analyze_size.py nabirds data/crops/nabirds/boxes.csv output nabirds_raw nabirds_dinov3_sq nabirds_gt_sq
主体大小 = 官方 bbox 面积 / 图面积（boxes.csv 的 gx*,W,H），分成 5 个等频组。"""
import csv, gzip, os, sys
import numpy as np

ds, boxes, pdir, *names = sys.argv[1:]
frac = {}
for r in csv.DictReader(gzip.open(boxes, "rt") if boxes.endswith(".gz") else open(boxes)):
    if r["split"] != "test" or not r["gx1"]:
        continue
    key = (r["cls"], os.path.basename(r["path"]))
    a = (float(r["gx2"]) - float(r["gx1"])) * (float(r["gy2"]) - float(r["gy1"]))
    frac[key] = a / (float(r["W"]) * float(r["H"]))

runs = {}
for n in names:
    z = np.load(os.path.join(pdir, f"{n}_preds.npz"))
    ok = (z["preds"] == z["labels"])
    if "paths" in z.files:                       # train_vit.py 直接产出的 npz 自带路径
        paths = list(z["paths"])
    else:                                        # results/ 里归档的 npz 去掉了路径，同一数据集共用一份 test_paths_<ds>.txt
        paths = open(os.path.join(pdir, f"test_paths_{ds}.txt")).read().split()
    keys = [(p.replace("\\", "/").split("/")[1], os.path.basename(p)) for p in paths]
    runs[n] = dict(zip(keys, ok))
keys = [k for k in runs[names[0]] if k in frac and all(k in runs[n] for n in names)]
f = np.array([frac[k] for k in keys])
edges = np.quantile(f, [0, .2, .4, .6, .8, 1.0])
print(f"[{ds}] n={len(keys)}  主体面积占比分位边界: " + " ".join(f"{e:.2f}" for e in edges))
print("组(面积占比)      n   " + "  ".join(f"{n:>16s}" for n in names))
for i in range(5):
    m = (f >= edges[i]) & (f <= edges[i + 1] if i == 4 else f < edges[i + 1])
    accs = [np.mean([runs[n][k] for k, mm in zip(keys, m) if mm]) * 100 for n in names]
    print(f"{edges[i]:.2f}–{edges[i+1]:.2f}   {m.sum():6d}   " + "  ".join(f"{a:16.2f}" for a in accs))
print("all               " + f"{len(keys):6d}   " + "  ".join(f"{np.mean(list(runs[n].values()))*100:16.2f}" for n in names))
