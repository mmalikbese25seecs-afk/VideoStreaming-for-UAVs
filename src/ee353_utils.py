"""
ee353_utils.py  --  shared helper code for our EE-353 project
=============================================================
Project: Victim-Recall-Aware Adaptive Video Streaming for Maritime UAV SAR

Every notebook does:   from ee353_utils import *
so both teammates use EXACTLY the same functions and settings.

The file is organised in small sections. Each function has a short
explanation so you can explain it in the viva:

  1. SETTINGS            - numbers we can change in one place
  2. DATASET HELPERS     - read SeaDronesSee, build a small subset
  3. METRICS             - did the detector find the swimmers? (recall)
  4. COMPRESSION         - JPEG encoding (uniform + region-priority)
  5. NETWORK SIMULATOR   - bandwidth budget + packet loss
  6. PRIORITY METHOD     - our proposed improvement
  7. EXPERIMENT RUNNER   - run one pipeline over many images
  8. GIT / COLAB HELPERS - push results, wait for teammate
"""

import io, os, json, math, glob, time, shutil, random, subprocess
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None   # SeaDronesSee images are large (5456 x 3632)

# =====================================================================
# 1. SETTINGS  (change here, everything else follows)
# =====================================================================
WORK_SIZE = 1280      # every image is resized so its LONG side = 1280 px
IMGSZ     = 1280      # image size the YOLO detector works at
CONF      = 0.25      # detector confidence needed to count as "a detection"
CAND_CONF = 0.05      # lower threshold used by the UAV to find CANDIDATE regions
IOU_THR   = 0.30      # overlap needed to say "detection matches ground truth"
MTU       = 1200      # bytes per network packet
STRIP_H   = 128       # image is sent as horizontal strips of this height
POS_BYTES = 8         # extra bytes to say WHERE a piece belongs in the image
ROI_MARGIN = 12       # pixels of context added around every priority region
FRAME_PERIOD_S = 1.0  # we send 1 frame per second -> budget = bandwidth * 1 s
DELAY_MS  = 200       # one-way link delay (only used when reporting latency)

# groups of swimmer sizes (side length in pixels at WORK_SIZE) for the report
SIZE_BINS = [(0, 16, "tiny (<16px)"), (16, 32, "small (16-32px)"), (32, 10**6, "medium (>32px)")]

DRIVE_DIR = "/content/drive/MyDrive/EE353_project"   # shared Google Drive folder
REPO_DIR  = "/content/repo"                          # where the GitHub repo is cloned
LOCAL_SUBSET = "/content/subset"                     # fast local copy of the data

IMG_EXT = {".jpg", ".jpeg", ".png"}


# =====================================================================
# 2. DATASET HELPERS
# =====================================================================
def explore_folder(root, depth=3):
    """Print the folder tree + file types so we can SEE how the dataset is organised."""
    from collections import Counter
    ext = Counter()
    for dp, dn, fn in os.walk(root):
        for f in fn:
            ext[os.path.splitext(f)[1].lower()] += 1
    print("File types found:", dict(ext))
    base = root.rstrip("/").count("/")
    for dp, dn, fn in os.walk(root):
        d = dp.count("/") - base
        if d <= depth:
            print("  " * d + os.path.basename(dp) + f"/   ({len(fn)} files)")
        if d >= depth:
            dn[:] = []          # do not go deeper


def list_images(root):
    return [p for p in glob.glob(root + "/**/*", recursive=True)
            if os.path.splitext(p)[1].lower() in IMG_EXT]


def guess_split(path, root):
    """Look at the folder names to guess: train / val / test."""
    rel = os.path.relpath(path, root).lower().replace("\\", "/")
    for part in rel.split("/")[:-1] + [os.path.splitext(rel.split("/")[-1])[0]]:
        if part.startswith("train"):
            return "train"
        if part.startswith("val"):
            return "val"
        if part.startswith("test"):
            return "test"
    return None


def load_annotations(root):
    """
    Read the dataset's labels. Works for the two common formats:
      * COCO json   (boxes as [x, y, w, h] in pixels)
      * YOLO txt    (one .txt per image: class cx cy w h, all between 0 and 1)
    Returns:
      records     - list of dicts: path, split, boxes=[(x1,y1,x2,y2,class_name), ...]
      class_names - dict  id -> name   (may be empty for YOLO without yaml)
    """
    all_imgs = list_images(root)
    by_name = {}
    for p in all_imgs:
        by_name.setdefault(os.path.basename(p), []).append(p)

    records, class_names = [], {}

    # ---- try COCO json first ----
    coco_found = False
    for jp in glob.glob(root + "/**/*.json", recursive=True):
        try:
            d = json.load(open(jp))
        except Exception:
            continue
        if not (isinstance(d, dict) and "images" in d and "annotations" in d):
            continue
        coco_found = True
        cats = {c["id"]: c["name"] for c in d.get("categories", [])}
        class_names.update(cats)
        json_split = guess_split(jp, root)
        boxes_by_img = {}
        for a in d["annotations"]:
            x, y, w, h = a["bbox"]
            boxes_by_img.setdefault(a["image_id"], []).append(
                (x, y, x + w, y + h, cats.get(a["category_id"], str(a["category_id"]))))
        for im in d["images"]:
            cands = by_name.get(os.path.basename(im["file_name"]), [])
            if not cands:
                continue
            # if the same file name exists in several folders, prefer the one in the json's split
            pick = [c for c in cands if guess_split(c, root) == json_split] or cands
            path = pick[0]
            records.append(dict(path=path, split=json_split or guess_split(path, root),
                                boxes=boxes_by_img.get(im["id"], [])))
    if coco_found:
        return records, class_names

    # ---- otherwise YOLO txt ----
    for yp in glob.glob(root + "/**/*.yaml", recursive=True):
        try:
            import yaml
            y = yaml.safe_load(open(yp))
            names = y.get("names") if isinstance(y, dict) else None
            if isinstance(names, list):
                class_names = {i: n for i, n in enumerate(names)}
            elif isinstance(names, dict):
                class_names = {int(k): v for k, v in names.items()}
            if class_names:
                break
        except Exception:
            pass

    for p in all_imgs:
        stem = os.path.splitext(p)[0]
        cands = [stem + ".txt",
                 stem.replace("/images/", "/labels/") + ".txt",
                 stem.replace("\\images\\", "\\labels\\") + ".txt"]
        lab = next((c for c in cands if os.path.exists(c)), None)
        if lab is None:
            continue
        w, h = Image.open(p).size
        boxes = []
        for line in open(lab).read().strip().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            c, cx, cy, bw, bh = int(float(parts[0])), *map(float, parts[1:5])
            boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h, class_names.get(c, str(c))))
        records.append(dict(path=p, split=guess_split(p, root), boxes=boxes))
    return records, class_names


def auto_person_classes(names):
    """Pick classes that are people in the water: names with 'swimmer' or 'floater' (but not 'boat')."""
    out = set()
    for n in names:
        s = str(n).lower()
        if ("swimmer" in s or "floater" in s) and "boat" not in s:
            out.add(n)
    return out


def build_subset(records, person_names, out_dir, n_train=1000, n_val=250,
                 seed=0, work_size=WORK_SIZE, progress=True):
    """
    Make a SMALL dataset that fits Colab:
      * keep only images that contain at least one person (swimmer / floater)
      * resize so the long side = work_size
      * save images + YOLO labels with ONE class: 0 = person
    """
    rng = random.Random(seed)
    has_person = lambda r: any(b[4] in person_names for b in r["boxes"])
    train_pool = [r for r in records if r["split"] in ("train", None) and has_person(r)]
    val_pool = [r for r in records if r["split"] == "val" and has_person(r)]
    if not val_pool:                       # no separate val folder -> split train 80/20
        rng.shuffle(train_pool)
        cut = max(1, int(0.2 * len(train_pool)))
        val_pool, train_pool = train_pool[:cut], train_pool[cut:]
    rng.shuffle(train_pool)
    rng.shuffle(val_pool)
    chosen = {"train": train_pool[:n_train], "val": val_pool[:n_val]}
    print(f"Pool sizes -> train: {len(train_pool)}, val: {len(val_pool)}  "
          f"| using train: {len(chosen['train'])}, val: {len(chosen['val'])}")

    for split, recs in chosen.items():
        os.makedirs(f"{out_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{out_dir}/labels/{split}", exist_ok=True)

    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = lambda x, **k: x

    n_boxes = {"train": 0, "val": 0}
    for split, recs in chosen.items():
        for i, r in enumerate(tqdm(recs, desc=f"building {split}", disable=not progress)):
            img = Image.open(r["path"]).convert("RGB")
            w, h = img.size
            s = min(1.0, work_size / max(w, h))
            nw, nh = max(1, round(w * s)), max(1, round(h * s))
            img = img.resize((nw, nh), Image.LANCZOS)
            name = f"{split}_{i:05d}"
            img.save(f"{out_dir}/images/{split}/{name}.jpg", quality=95)
            lines = []
            for (x1, y1, x2, y2, cls) in r["boxes"]:
                if cls not in person_names:
                    continue
                x1, x2 = np.clip([x1 * s, x2 * s], 0, nw)
                y1, y2 = np.clip([y1 * s, y2 * s], 0, nh)
                if (x2 - x1) < 1.5 or (y2 - y1) < 1.5:
                    continue
                lines.append(f"0 {(x1+x2)/2/nw:.6f} {(y1+y2)/2/nh:.6f} {(x2-x1)/nw:.6f} {(y2-y1)/nh:.6f}")
            open(f"{out_dir}/labels/{split}/{name}.txt", "w").write("\n".join(lines))
            n_boxes[split] += len(lines)

    open(f"{out_dir}/data.yaml", "w").write(
        f"path: {out_dir}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n")
    print("Persons (boxes) saved ->", n_boxes)
    return n_boxes


def load_items(subset_dir, split="val"):
    """Load image paths + ground-truth boxes (pixels, x1 y1 x2 y2) for one split."""
    items = []
    for p in sorted(glob.glob(f"{subset_dir}/images/{split}/*.jpg")):
        lab = p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        w, h = Image.open(p).size
        gt = []
        if os.path.exists(lab):
            for line in open(lab).read().strip().splitlines():
                c, cx, cy, bw, bh = map(float, line.split())
                gt.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        items.append(dict(path=p, gt=np.array(gt, dtype=float).reshape(-1, 4)))
    return items


def split_tune_test(items):
    """
    Fair testing: half of the validation images are used to TUNE our method
    (Person A), the other half is only used for the FINAL comparison (Person B).
    The shuffle is fixed (seed 42) so both teammates get the same halves.
    """
    idx = list(range(len(items)))
    random.Random(42).shuffle(idx)
    half = len(idx) // 2
    return [items[i] for i in idx[:half]], [items[i] for i in idx[half:]]


# =====================================================================
# 3. METRICS
# =====================================================================
def box_iou(a, b):
    """IoU (overlap score 0..1) between every box in a (N x 4) and every box in b (M x 4)."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_gt(gt, pred, iou_thr=IOU_THR):
    """
    For every ground-truth person: was it found?  (True / False)
    pred rows are [x1, y1, x2, y2, confidence]. Most confident detections are matched first.
    """
    found = np.zeros(len(gt), dtype=bool)
    if len(gt) == 0 or len(pred) == 0:
        return found
    ious = box_iou(gt, pred[:, :4])
    for pi in np.argsort(-pred[:, 4]):
        col = ious[:, pi].copy()
        col[found] = 0                      # a GT box can only be found once
        g = col.argmax()
        if col[g] >= iou_thr:
            found[g] = True
    return found


def detect(model, img, conf=CONF, imgsz=IMGSZ):
    """Run YOLO on one PIL image -> array of [x1, y1, x2, y2, confidence] (class 0 = person only)."""
    r = model.predict(img, imgsz=imgsz, conf=conf, verbose=False, classes=[0])[0]
    if len(r.boxes) == 0:
        return np.zeros((0, 5))
    return np.concatenate([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()[:, None]], axis=1)


# =====================================================================
# 4. COMPRESSION  (JPEG, the simplest codec we can fully control)
# =====================================================================
def jpeg_encode(img, quality):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=int(quality))
    return buf.getvalue()


def jpeg_decode(data):
    return Image.open(io.BytesIO(data)).convert("RGB")


@lru_cache(maxsize=None)
def _jpeg_header_bytes(quality):
    """Size of a JPEG file header. A real video codec sends ONE header per stream,
    not one per piece, so we do not charge it for every strip/crop (fair for both methods)."""
    return len(jpeg_encode(Image.new("RGB", (8, 8)), quality)) - 10


def unit_bytes(unit):
    """Bytes we charge for one piece ('unit') of the picture."""
    return max(len(unit["data"]) - _jpeg_header_bytes(unit["quality"]), 1) + POS_BYTES


def total_bytes(units):
    return sum(unit_bytes(u) for u in units)


def encode_uniform(img, quality, strip_h=STRIP_H):
    """BASELINE: the whole picture gets the SAME quality. Sent as horizontal strips."""
    W, H = img.size
    units = []
    for y0 in range(0, H, strip_h):
        strip = img.crop((0, y0, W, min(H, y0 + strip_h)))
        units.append(dict(kind="strip", y0=y0, quality=quality, data=jpeg_encode(strip, quality)))
    return units


def encode_roi(img, boxes, q_bg, q_roi=90, margin=ROI_MARGIN):
    """OURS: background strips at LOW quality + small crops around priority boxes at HIGH quality."""
    W, H = img.size
    units = encode_uniform(img, q_bg)
    for (x1, y1, x2, y2) in boxes:
        x1, y1 = int(max(0, x1 - margin)), int(max(0, y1 - margin))
        x2, y2 = int(min(W, x2 + margin)), int(min(H, y2 + margin))
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        crop = img.crop((x1, y1, x2, y2))
        units.append(dict(kind="roi", box=(x1, y1, x2, y2), quality=q_roi, data=jpeg_encode(crop, q_roi)))
    return units


# =====================================================================
# 5. NETWORK SIMULATOR
# =====================================================================
def budget_from_bandwidth(bw_kbps, period_s=FRAME_PERIOD_S):
    """How many bytes can we send per frame on a link of bw_kbps?"""
    return bw_kbps * 1000 / 8 * period_s


def latency_s(nbytes, bw_kbps, delay_ms=DELAY_MS):
    """Time for the frame to arrive = transmission time + link delay."""
    return nbytes * 8 / (bw_kbps * 1000) + delay_ms / 1000


def receive(units, size, loss_rate, rng):
    """
    Pretend to send the units over a lossy link and rebuild the picture at the ground station.
      * every unit is cut into packets of MTU bytes
      * every packet is lost independently with probability loss_rate (no retransmission, like UDP)
      * a unit with ANY lost packet is thrown away
      * missing strips stay grey; missing ROI crops just stay at background quality
    Returns (received_image, bytes_sent, fraction_of_units_lost)
    """
    canvas = Image.new("RGB", size, (114, 114, 114))
    sent, lost = 0, 0
    ordered = [u for u in units if u["kind"] == "strip"] + [u for u in units if u["kind"] == "roi"]
    for u in ordered:
        nb = unit_bytes(u)
        sent += nb
        n_packets = math.ceil(nb / MTU)
        if rng.random() < (1 - loss_rate) ** n_packets:       # survived?
            pic = jpeg_decode(u["data"])
            if u["kind"] == "strip":
                canvas.paste(pic, (0, u["y0"]))
            else:
                canvas.paste(pic, u["box"][:2])
        else:
            lost += 1
    return canvas, sent, lost / max(1, len(ordered))


# =====================================================================
# 6. PRIORITY METHOD  (our proposed improvement)
# =====================================================================
def priority_scores(cands, alpha=0.5):
    """
    Score from our proposal:   P_k = alpha * (1 - c_k) + (1 - alpha) * (1 - A_k / A_max)
      c_k   = detector confidence of candidate k      (uncertain  -> higher score)
      A_k   = box area, A_max = biggest candidate     (small      -> higher score)
    """
    if len(cands) == 0:
        return np.zeros(0)
    area = (cands[:, 2] - cands[:, 0]) * (cands[:, 3] - cands[:, 1])
    a_max = max(area.max(), 1.0)
    return alpha * (1 - cands[:, 4]) + (1 - alpha) * (1 - area / a_max)


def pick_regions(cands, alpha=0.5, k_max=20):
    """Return the k_max highest-priority boxes, best first."""
    if len(cands) == 0:
        return np.zeros((0, 4))
    order = np.argsort(-priority_scores(cands, alpha))[:k_max]
    return cands[order, :4]


def fit_uniform(img, budget_bytes, q_min=5, q_max=95):
    """Baseline 'adaptive bitrate': highest JPEG quality that still fits the byte budget (binary search)."""
    lo, hi, best = q_min, q_max, q_min
    while lo <= hi:
        mid = (lo + hi) // 2
        if total_bytes(encode_uniform(img, mid)) <= budget_bytes:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def fit_roi(img, cands, budget_bytes, alpha=0.5, k_max=20, q_roi=90, q_min=5, q_max=90):
    """
    OUR 'adaptive' method: keep priority crops sharp, give the background whatever bytes remain.
    If even the lowest background quality does not fit, drop the least important crops.
    Returns (units, q_bg, number_of_crops_used)
    """
    boxes = pick_regions(cands, alpha, k_max)
    k = len(boxes)
    while k > 0 and total_bytes(encode_roi(img, boxes[:k], q_min, q_roi)) > budget_bytes:
        k -= 1
    lo, hi, best = q_min, q_max, q_min
    while lo <= hi:
        mid = (lo + hi) // 2
        if total_bytes(encode_roi(img, boxes[:k], mid, q_roi)) <= budget_bytes:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return encode_roi(img, boxes[:k], best, q_roi), best, k


def get_candidates(model, items, conf=CAND_CONF, progress=True):
    """UAV side: run the detector ONCE on every original frame to find possible swimmers."""
    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = lambda x, **k: x
    for it in tqdm(items, desc="finding candidates", disable=not progress):
        it["cands"] = detect(model, Image.open(it["path"]).convert("RGB"), conf=conf)
    return items


# =====================================================================
# 7. EXPERIMENT RUNNER
# =====================================================================
# A "pipeline" is a function:  pipeline(img, item, rng) -> (received_img, bytes_sent, extra_number)

def make_identity_pipeline():
    """No compression at all (reference). Bytes = size of the stored file."""
    def run(img, item, rng):
        return img, os.path.getsize(item["path"]), 0
    return run


def make_uniform_pipeline(quality=None, budget_bytes=None, loss=0.0):
    """Baseline. Give a fixed JPEG quality OR a byte budget (then quality is chosen automatically)."""
    def run(img, item, rng):
        q = quality if budget_bytes is None else fit_uniform(img, budget_bytes)
        out, sent, _ = receive(encode_uniform(img, q), img.size, loss, rng)
        return out, sent, q
    return run


def make_roi_pipeline(budget_bytes, alpha=0.5, k_max=20, q_roi=90, loss=0.0):
    """Our method. Needs item['cands'] (see get_candidates)."""
    def run(img, item, rng):
        units, q_bg, k = fit_roi(img, item["cands"], budget_bytes, alpha, k_max, q_roi)
        out, sent, _ = receive(units, img.size, loss, rng)
        return out, sent, q_bg
    return run


def run_experiment(model, items, pipeline, name="", seed=0, conf=CONF, progress=True):
    """
    Send every image through the pipeline, run the detector on what arrives,
    and count how many ground-truth swimmers were still found (RECALL).
    """
    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = lambda x, **k: x
    rng = np.random.default_rng(seed)
    n_bins = len(SIZE_BINS)
    found_bin, gt_bin = np.zeros(n_bins), np.zeros(n_bins)
    n_pred = tp = 0
    kbs, extras = [], []
    for it in tqdm(items, desc=name, disable=not progress, leave=False):
        img = Image.open(it["path"]).convert("RGB")
        out, nbytes, extra = pipeline(img, it, rng)
        pred = detect(model, out, conf=conf)
        found = match_gt(it["gt"], pred)
        sides = np.sqrt(np.clip(it["gt"][:, 2] - it["gt"][:, 0], 0, None) *
                        np.clip(it["gt"][:, 3] - it["gt"][:, 1], 0, None)) if len(it["gt"]) else np.zeros(0)
        for b, (lo, hi, _) in enumerate(SIZE_BINS):
            m = (sides >= lo) & (sides < hi)
            gt_bin[b] += m.sum()
            found_bin[b] += (found & m).sum()
        n_pred += len(pred)
        tp += found.sum()
        kbs.append(nbytes / 1000)
        extras.append(extra)
    n_gt = gt_bin.sum()
    res = dict(name=name, recall=found_bin.sum() / max(n_gt, 1), precision=tp / max(n_pred, 1),
               mean_kb=float(np.mean(kbs)), mean_param=float(np.mean(extras)), n_gt=int(n_gt))
    for b, (_, _, label) in enumerate(SIZE_BINS):
        res["recall_" + label.split()[0]] = found_bin[b] / max(gt_bin[b], 1)
    return res


# =====================================================================
# 8. DRAWING + GIT / COLAB HELPERS
# =====================================================================
def draw_boxes(img, boxes, color="lime", width=2):
    """Return a copy of img with rectangles drawn (for figures). boxes: N x 4 (or N x 5)."""
    out = img.copy()
    d = ImageDraw.Draw(out)
    for b in np.asarray(boxes, float).reshape(len(boxes), -1)[:, :4]:
        d.rectangle([float(v) for v in b], outline=color, width=width)
    return out


def _git(*args, cwd=None):
    r = subprocess.run(["git", *args], cwd=cwd or REPO_DIR, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def git_push(message, paths):
    """
    Save our results to GitHub:  add -> commit -> pull (get teammate's work) -> push.
    paths = list of files/folders relative to the repo, e.g. ["results", "models/detector_best.pt"]
    """
    _git("add", *paths)
    code, out = _git("commit", "-m", message)
    if code != 0 and "nothing to commit" in out:
        print("Nothing new to commit.")
    code, out = _git("pull", "--rebase", "--autostash")
    if code != 0:
        print("PULL PROBLEM:\n", out); return False
    code, out = _git("push")
    print("PUSHED OK  ->", message) if code == 0 else print("PUSH PROBLEM:\n", out)
    return code == 0


def wait_for(paths, every=30, timeout_min=240):
    """
    Wait until the teammate has finished their part.
    Relative paths are looked up inside the repo (we 'git pull' every time),
    absolute paths (e.g. on Google Drive) are checked directly.
    """
    t0 = time.time()
    resolve = lambda p: p if os.path.isabs(p) else os.path.join(REPO_DIR, p)
    while True:
        _git("pull", "--rebase", "--autostash")
        missing = [p for p in paths if not os.path.exists(resolve(p))]
        if not missing:
            print("GO!  Everything you were waiting for is here:", paths)
            return True
        if (time.time() - t0) / 60 > timeout_min:
            print("Gave up waiting for:", missing); return False
        print(f"[{time.strftime('%H:%M:%S')}] still waiting for: {missing}")
        time.sleep(every)
