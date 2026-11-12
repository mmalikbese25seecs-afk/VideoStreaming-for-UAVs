import sys, os, glob
import pandas as pd
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from common import *

ROOT = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 150
CRFS = [18, 24, 30, 36, 42, 48]

model = YOLO("yolov8s.pt")
cls = person_class_id(ROOT)
img_dir, lbl_dir = split_dirs(ROOT, "valid")
paths = sorted(glob.glob(f"{img_dir}/*.jpg") + glob.glob(f"{img_dir}/*.png"))

rows, used = [], 0
for p in tqdm(paths):
    if used >= N:
        break
    w, h = Image.open(p).size
    stem = os.path.splitext(os.path.basename(p))[0]
    gt = load_gt(os.path.join(lbl_dir, stem + ".txt"), w, h, cls)
    if len(gt) == 0:
        continue
    used += 1
    name = os.path.basename(p)
    pred, _ = detect(model, Image.open(p).convert("RGB"))
    rows.append(dict(image=name, crf="orig", bytes=os.path.getsize(p),
                     n_gt=len(gt), hits=count_hits(gt, pred)))
    for crf in CRFS:
        img, nb = encode_x264(p, crf)
        pred, _ = detect(model, img)
        rows.append(dict(image=name, crf=str(crf), bytes=nb,
                         n_gt=len(gt), hits=count_hits(gt, pred)))

os.makedirs("results", exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv("results/baseline_x264.csv", index=False)
s = df.groupby("crf", sort=False).agg(mean_KB=("bytes", lambda x: x.mean()/1024),
                                       hits=("hits", "sum"), gt=("n_gt", "sum"))
s["recall"] = s.hits / s.gt
print(s)
