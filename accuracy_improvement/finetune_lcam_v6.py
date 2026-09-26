"""
LCAM-YOLOX v6 Fine-Tuning Script (50 epochs) -- IMPROVED RECIPE

Same starting point and epoch budget as the v5 script that produced the
currently-deployed model (fine-tunes from v3.2 weights, 50 epochs, batch 16,
SGD lr=1e-4, cosine schedule) so this isolates the effect of the RECIPE,
not a confound like "more epochs" or "different starting weights".

Four changes from v5, each addressing a gap found by reading the actual
v5 script that ran on Kaggle (notebook9249dbff4e, 2026-07-06):
  1. Real validation each epoch: val LOSS every epoch, val mAP@0.5 every
     `map_eval_freq` epochs (via YOLOX's own postprocess + a self-contained
     IoU-matching AP, no hardware/xir dependency -- runs fine on Kaggle).
     Best checkpoint is now selected by VALIDATION mAP, not training loss.
  2. Mixup + HSV colour jitter added to the custom DFireDataset (kept
     hand-rolled rather than switching to YOLOX's COCO-format Mosaic
     pipeline, since the dataset is YOLO-txt format and a format migration
     is a bigger, separate change not worth conflating with this run).
  3. EMA (exponential moving average) of weights, YOLOX-style; the EMA
     model is what gets evaluated and saved as "best".
  4. Explicit --seed CLI arg (torch/numpy/random all seeded) so this can be
     invoked several times for a multi-seed mean+std report.

Usage on Kaggle (same pattern as the v5 run):
  - Upload this folder (with the v5 YOLOX/yolox/models/lcam.py, UNCHANGED)
    as a dataset
  - Attach the D-Fire dataset (smoke-fire-detection-yolo)
  - Attach the v3.2 checkpoint as the starting point (same as v5)
  - Run in a Kaggle notebook with GPU T4 enabled, e.g.:
      python finetune_lcam_v6.py --seed 0
"""

import argparse
import copy
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "/kaggle/working/YOLOX")

print("=" * 70)
print("LCAM-YOLOX v6 Fine-Tuning (validation-selected, mixup+HSV, EMA)")
print("=" * 70)

CLI = argparse.ArgumentParser()
CLI.add_argument("--seed", type=int, default=0)
CLI.add_argument("--epochs", type=int, default=50)
CLI.add_argument("--out-suffix", default="")
ARGS = CLI.parse_args()

CONFIG = {
    "dataset_root": "/kaggle/input/smoke-fire-detection-yolo/data",
    # INPUT: v3.2 weights -- SAME starting point as the v5 run, so this
    # experiment isolates the training-recipe change, nothing else.
    "checkpoint_path": "/kaggle/working/lcam_yolox_wildfire_v32_compat.pth",
    "output_dir": "/kaggle/working/output_v6%s" % (
        ("_" + ARGS.out_suffix) if ARGS.out_suffix else ""),
    "depth": 0.33, "width": 0.5, "num_classes": 2, "input_size": 640,
    "epochs": ARGS.epochs,
    "batch_size": 16,
    "learning_rate": 1e-4,
    "weight_decay": 5e-4,
    "momentum": 0.9,
    "num_workers": 4,
    "print_freq": 50,
    "save_freq": 5,
    "map_eval_freq": 5,       # full val-mAP pass every N epochs (slower)
    "mixup_prob": 0.5,
    "mixup_beta": 8.0,        # YOLOX-style Beta(alpha, alpha) mixing ratio
    "hsv_prob": 0.5,
    "ema_decay": 0.9998,
    "conf_thre": 0.05,        # for val-mAP decode (low floor, matches
    "nms_thre": 0.45,         # eval_map.py's convention on the board)
    "seed": ARGS.seed,
}

print("\nConfiguration:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =====================================================================
# DATASET -- D-Fire YOLO format, now with mixup + HSV jitter
# =====================================================================
def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    """Standard YOLOX-style HSV jitter, in place on a uint8 HWC RGB image."""
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    dtype = img.dtype
    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
    merged = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    return cv2.cvtColor(merged, cv2.COLOR_HSV2RGB)


class DFireDataset(Dataset):
    """D-Fire YOLO format: images/IMG.jpg, labels/IMG.txt (cls x y w h normalized).
    Adds mixup (blend two images + concatenate boxes) and HSV jitter on top
    of the v5 script's flip-only augmentation, train split only."""

    def __init__(self, root, split="train", input_size=640, augment=True,
                 mixup_prob=0.0, mixup_beta=8.0, hsv_prob=0.0):
        self.root = Path(root)
        self.split = split
        self.input_size = input_size
        self.augment = augment and (split == "train")
        self.mixup_prob = mixup_prob if self.augment else 0.0
        self.mixup_beta = mixup_beta
        self.hsv_prob = hsv_prob if self.augment else 0.0
        self.img_dir = self.root / split / "images"
        self.lbl_dir = self.root / split / "labels"
        self.images = sorted(list(self.img_dir.glob("*.jpg")))
        print(f"  {split}: {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def load_labels(self, img_path):
        lbl_path = self.lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            return np.zeros((0, 5), dtype=np.float32)
        labels = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                    labels.append([cls, x, y, w, h])
        return np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)

    def _load_resized(self, idx):
        img_path = self.images[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Cannot read: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        labels = self.load_labels(img_path).copy()
        img = cv2.resize(img, (self.input_size, self.input_size))
        return img, labels

    def __getitem__(self, idx):
        img, labels = self._load_resized(idx)

        if self.mixup_prob and np.random.random() < self.mixup_prob:
            idx2 = np.random.randint(len(self.images))
            img2, labels2 = self._load_resized(idx2)
            r = float(np.random.beta(self.mixup_beta, self.mixup_beta))
            img = (img.astype(np.float32) * r +
                   img2.astype(np.float32) * (1.0 - r)).astype(np.uint8)
            labels = (np.concatenate([labels, labels2], axis=0)
                      if len(labels2) else labels)

        if self.hsv_prob and np.random.random() < self.hsv_prob:
            img = augment_hsv(img)

        if self.augment and np.random.random() > 0.5:
            img = img[:, ::-1, :].copy()
            if len(labels) > 0:
                labels[:, 1] = 1 - labels[:, 1]

        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img = torch.from_numpy(img)
        return img, labels


def collate_fn(batch):
    imgs, lbls = zip(*batch)
    imgs = torch.stack(imgs)
    max_n = max(max(len(l) for l in lbls), 1)
    padded = np.zeros((len(lbls), max_n, 5), dtype=np.float32)
    for i, l in enumerate(lbls):
        if len(l) > 0:
            padded[i, :len(l)] = l
    return imgs, torch.from_numpy(padded)


# =====================================================================
# EMA -- YOLOX-style exponential moving average of weights
# =====================================================================
class ModelEMA:
    def __init__(self, model, decay=0.9998):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v.copy_(v * self.decay + (1.0 - self.decay) * msd[k].detach())
                else:
                    v.copy_(msd[k])


# =====================================================================
# VALIDATION -- loss (cheap, every epoch) and mAP@0.5 (every N epochs)
# =====================================================================
def val_loss(model, val_loader, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            out = model(imgs, targets)
            total += out["total_loss"].item()
            n += 1
    return total / max(n, 1)


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = ((a[2] - a[0]) * (a[3] - a[1]) +
          (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def average_precision(dets, gts, cls, thr=0.5):
    """Same all-point-interpolation AP as the board's eval_map.py, kept
    dependency-free (no xir/vart) so it runs identically on Kaggle."""
    d = sorted([x for x in dets if x[1] == cls], key=lambda x: -x[2])
    npos = sum(1 for g in gts.values() for c, *_ in g if c == cls)
    if npos == 0:
        return None
    claimed = {}
    tp, fp = np.zeros(len(d)), np.zeros(len(d))
    for i, (img, _, _, box) in enumerate(d):
        best, bi = 0.0, -1
        for j, (c, *gb) in enumerate(gts.get(img, [])):
            if c != cls or claimed.get((img, j)):
                continue
            v = iou_xyxy(box, gb)
            if v > best:
                best, bi = v, j
        if best >= thr and bi >= 0:
            claimed[(img, bi)] = True
            tp[i] = 1
        else:
            fp[i] = 1
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / npos
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for k in range(len(mpre) - 2, -1, -1):
        mpre[k] = max(mpre[k], mpre[k + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def val_map(model, val_ds, device, num_classes, conf_thre, nms_thre, input_size=640):
    from yolox.utils import postprocess
    model.eval()
    dets, gts = [], {}
    with torch.no_grad():
        for i in range(len(val_ds)):
            img_path = val_ds.images[i]
            key = img_path.stem
            raw = cv2.imread(str(img_path))
            oh, ow = raw.shape[:2]
            gt_raw = val_ds.load_labels(img_path)
            gts[key] = [(int(c), (cx - w / 2) * ow, (cy - h / 2) * oh,
                         (cx + w / 2) * ow, (cy + h / 2) * oh)
                        for c, cx, cy, w, h in gt_raw]

            img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (input_size, input_size))
            x = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
            out = postprocess(model(x), num_classes, conf_thre, nms_thre)[0]
            if out is not None:
                out = out.cpu().numpy()
                sx, sy = ow / input_size, oh / input_size
                for det in out:
                    x1, y1, x2, y2 = det[:4]
                    score = float(det[4] * det[5])
                    cls = int(det[6])
                    dets.append((key, cls, score,
                                 (x1 * sx, y1 * sy, x2 * sx, y2 * sy)))
    aps = [average_precision(dets, gts, c, 0.5) for c in range(num_classes)]
    aps = [a for a in aps if a is not None]
    return float(np.mean(aps)) if aps else 0.0


# =====================================================================
# MAIN
# =====================================================================
def main():
    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}  seed: {CONFIG['seed']}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    print("\n[1/4] Building LCAM-YOLOX v6 model (v5 architecture, unchanged)...")
    from yolox.exp import get_exp
    exp = get_exp(exp_file=None, exp_name="yolox_s")
    exp.depth = CONFIG["depth"]
    exp.width = CONFIG["width"]
    exp.num_classes = CONFIG["num_classes"]
    exp.input_size = (CONFIG["input_size"], CONFIG["input_size"])
    exp.act = "lrelu"
    model = exp.get_model()
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")

    from yolox.models.lcam import ChannelAttention
    import inspect
    ca_src = inspect.getsource(ChannelAttention)
    has_hardsig = "Hardsigmoid" in ca_src
    has_staged = "StagedGlobalAvgPool" in inspect.getsource(sys.modules["yolox.models.lcam"])
    print(f"  v5 architecture check -> Hardsigmoid: {has_hardsig}, StagedGlobalAvgPool: {has_staged}")
    if not (has_hardsig and has_staged):
        print("  WARNING: v5 architecture NOT detected! Check lcam.py is the v5 version.")

    print("\n[2/4] Loading v3.2 checkpoint for fine-tuning (same starting point as v5)...")
    ckpt = torch.load(CONFIG["checkpoint_path"], map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Loaded (strict=False). Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    model = model.to(device)
    ema = ModelEMA(model, decay=CONFIG["ema_decay"])

    print("\n[3/4] Setting up data (train: mixup+HSV+flip, val: none)...")
    train_ds = DFireDataset(CONFIG["dataset_root"], split="train",
                             input_size=CONFIG["input_size"], augment=True,
                             mixup_prob=CONFIG["mixup_prob"],
                             mixup_beta=CONFIG["mixup_beta"],
                             hsv_prob=CONFIG["hsv_prob"])
    val_ds = DFireDataset(CONFIG["dataset_root"], split="val",
                           input_size=CONFIG["input_size"], augment=False)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
                               num_workers=CONFIG["num_workers"], collate_fn=collate_fn,
                               pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
                             num_workers=CONFIG["num_workers"], collate_fn=collate_fn,
                             pin_memory=True)
    print(f"  Train batches: {len(train_loader)}   Val batches: {len(val_loader)}")

    optimizer = optim.SGD(model.parameters(), lr=CONFIG["learning_rate"],
                           momentum=CONFIG["momentum"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    print(f"\n[4/4] Starting fine-tuning ({CONFIG['epochs']} epochs, seed={CONFIG['seed']})...")
    history = []
    best_map = -1.0
    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        for i, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(imgs, targets)
            loss = outputs["total_loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(model)
            epoch_loss += loss.item()
            if (i + 1) % CONFIG["print_freq"] == 0:
                avg = epoch_loss / (i + 1)
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - epoch_start
                fps = (i + 1) * CONFIG["batch_size"] / elapsed
                print(f"  Epoch {epoch+1}/{CONFIG['epochs']} [{i+1}/{len(train_loader)}] "
                      f"loss={avg:.4f} lr={lr:.6f} fps={fps:.1f}")
        scheduler.step()
        train_avg = epoch_loss / len(train_loader)

        vloss = val_loss(ema.ema, val_loader, device)
        do_map = ((epoch + 1) % CONFIG["map_eval_freq"] == 0) or ((epoch + 1) == CONFIG["epochs"])
        vmap = (val_map(ema.ema, val_ds, device, CONFIG["num_classes"],
                         CONFIG["conf_thre"], CONFIG["nms_thre"], CONFIG["input_size"])
                if do_map else None)

        print(f"\nEpoch {epoch+1} complete: train_loss={train_avg:.4f} "
              f"val_loss={vloss:.4f} val_mAP@0.5={('%.4f' % vmap) if vmap is not None else 'skip'} "
              f"({time.time()-epoch_start:.1f}s)")
        history.append(dict(epoch=epoch + 1, train_loss=train_avg, val_loss=vloss, val_map=vmap))

        if (epoch + 1) % CONFIG["save_freq"] == 0 or (epoch + 1) == CONFIG["epochs"]:
            ckpt_path = os.path.join(CONFIG["output_dir"], f"v6_epoch_{epoch+1}.pth")
            torch.save({"model": ema.ema.state_dict(), "epoch": epoch + 1,
                        "train_loss": train_avg, "val_loss": vloss, "val_map": vmap,
                        "seed": CONFIG["seed"]}, ckpt_path)
            print(f"  Saved: {ckpt_path}")

        if vmap is not None and vmap > best_map:
            best_map = vmap
            best_path = os.path.join(CONFIG["output_dir"], "lcam_yolox_v6_best.pth")
            torch.save({"model": ema.ema.state_dict(), "epoch": epoch + 1,
                        "train_loss": train_avg, "val_loss": vloss, "val_map": vmap,
                        "seed": CONFIG["seed"]}, best_path)
            print(f"  New best (val_mAP@0.5={best_map:.4f}) -> {best_path}")

    final_path = os.path.join(CONFIG["output_dir"], "lcam_yolox_v6_final.pth")
    torch.save({"model": ema.ema.state_dict(), "epoch": CONFIG["epochs"],
                "val_map": best_map, "seed": CONFIG["seed"]}, final_path)

    import json
    with open(os.path.join(CONFIG["output_dir"], "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("v6 FINE-TUNING COMPLETE")
    print("=" * 70)
    print(f"Best val mAP@0.5: {best_map:.4f}")
    print(f"Best  weights: {os.path.join(CONFIG['output_dir'], 'lcam_yolox_v6_best.pth')}")
    print(f"Final weights: {final_path}")
    print(f"History:       {os.path.join(CONFIG['output_dir'], 'history.json')}")


if __name__ == "__main__":
    main()
