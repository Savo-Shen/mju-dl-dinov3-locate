"""抽样画框：红 = DINOv3 自动框，青 = 官方 bbox。用法: python locate/viz_boxes.py <ds> [crops_dir] → /tmp/<ds>_boxes.jpg"""
import csv, os, random, sys
from PIL import Image, ImageDraw
ds = sys.argv[1]
D = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("CROPS_DIR", "data/crops")
rows = [r for r in csv.DictReader(open(f"{D}/{ds}/boxes.csv")) if r["split"] == "test"]
random.seed(3); pick = random.sample(rows, 8)
if rows[0]["iou"]:
    pick += sorted(rows, key=lambda r: float(r["iou"]))[:2]
tiles = []
for r in pick:
    im = Image.open(r["path"]).convert("RGB"); d = ImageDraw.Draw(im)
    a = [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
    if r["gx1"]:
        d.rectangle([float(r[k]) for k in ("gx1", "gy1", "gx2", "gy2")], outline=(0, 255, 255), width=4)
    d.rectangle(a, outline=(255, 0, 0), width=4)
    im.thumbnail((300, 300)); t = Image.new("RGB", (300, 320), "white"); t.paste(im, (0, 0))
    ImageDraw.Draw(t).text((4, 302), "IoU " + (r["iou"][:4] if r["iou"] else "-"), fill="black"); tiles.append(t)
sheet = Image.new("RGB", (5 * 300, 2 * 320), "white")
for i, t in enumerate(tiles): sheet.paste(t, ((i % 5) * 300, (i // 5) * 320))
sheet.save(f"/tmp/{ds}_boxes.jpg", quality=85)
