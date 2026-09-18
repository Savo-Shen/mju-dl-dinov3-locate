# -*- coding: utf-8 -*-
"""NABirds 分析案例：小主体图里 原图预测错、DINOv3 裁剪后预测对 的例子，画 原图+自动框 | 裁剪图，标真实类/两种预测。"""
import csv, os, random, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 用法: python locate/nabirds_cases.py [nabirds 原始目录] [裁剪目录] [output 目录]
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("DATA_ROOT", "data"), "nabirds")
D = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.environ.get("CROPS_DIR", "data/crops"), "nabirds")
OUTDIR = sys.argv[3] if len(sys.argv) > 3 else "output"
OUT = "/tmp/nabirds_cases.jpg"
names = {}
for line in open(os.path.join(ROOT, "classes.txt")):
    i, n = line.rstrip("\n").split(" ", 1); names[int(i)] = n
classes = sorted(os.listdir(os.path.join(D, "raw", "test")))          # ImageFolder 的类索引顺序
cls_name = {k: names[int(c)] for k, c in enumerate(classes)}
box = {}
for r in csv.DictReader(open(os.path.join(D, "boxes_sq.csv"))):
    if r["split"] == "test":
        key = (r["cls"], os.path.basename(r["path"]))
        g = [float(r[k]) for k in ("gx1", "gy1", "gx2", "gy2")]
        box[key] = ([float(r[k]) for k in ("x1", "y1", "x2", "y2")], (g[2]-g[0])*(g[3]-g[1])/(float(r["W"])*float(r["H"])), r["path"])
def load(n):
    z = np.load(os.path.join(OUTDIR, f"{n}_preds.npz"))
    keys = [(p.split("/")[1], os.path.basename(p)) for p in z["paths"]]
    return dict(zip(keys, zip(z["preds"], z["labels"])))
raw, crop = load("nabirds_raw"), load("nabirds_dinov3_sq")
cands = [k for k in raw if k in crop and k in box and box[k][1] < 0.10 and raw[k][0] != raw[k][1] and crop[k][0] == crop[k][1]]
print("candidates", len(cands))
random.seed(7); random.shuffle(cands)
seen, pick = set(), []
for k in cands:
    if raw[k][1] in seen: continue
    seen.add(raw[k][1]); pick.append(k)
    if len(pick) == 4: break
font = None
for f in ["/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
    if os.path.exists(f): font = ImageFont.truetype(f, 16); break
H = 190; TXT = 62; tiles = []
for k in pick:
    b, frac, path = box[k]
    im = Image.open(path).convert("RGB"); d = ImageDraw.Draw(im)
    d.rectangle(b, outline=(255, 40, 40), width=max(3, im.width // 150))
    cr = Image.open(os.path.join(D, "dinov3_sq", "test", k[0], k[1])).convert("RGB")
    im = im.resize((int(im.width * H / im.height), H)); cr = cr.resize((H, H))
    t = Image.new("RGB", (im.width + cr.width + 12, H + TXT), "white"); t.paste(im, (0, 0)); t.paste(cr, (im.width + 12, 0))
    dd = ImageDraw.Draw(t)
    dd.text((0, H + 4), f"GT: {cls_name[int(raw[k][1])]}   (bird = {frac*100:.0f}% of image)", fill="black", font=font)
    dd.text((0, H + 22), f"raw:  {cls_name[int(raw[k][0])]}  X", fill=(200, 30, 30), font=font)
    dd.text((0, H + 40), f"crop: {cls_name[int(crop[k][0])]}  OK", fill=(20, 120, 60), font=font)
    tiles.append(t)
W = max(t.width for t in tiles); cols = 2; rows = (len(tiles) + 1) // 2
sheet = Image.new("RGB", (cols * W + 24, rows * (H + TXT) + 24), "white")
for i, t in enumerate(tiles): sheet.paste(t, (12 + (i % cols) * W, 12 + (i // cols) * (H + TXT)))
sheet.save(OUT, quality=90); print(OUT, sheet.size)
for k in pick: print(k, cls_name[int(raw[k][1])], "| raw:", cls_name[int(raw[k][0])], "| frac", round(box[k][1],3))
