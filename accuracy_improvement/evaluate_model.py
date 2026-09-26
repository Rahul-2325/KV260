"""
Paper-style evaluation of a trained LCAM-YOLOX checkpoint (FP32, PyTorch).

Produces, per split (val and/or test):
  - COCO metrics (pycocotools): AP, AP50, AP75, AP_S/M/L, AR_1/10/100
  - per-class AP and AP50
  - precision / recall / F1 per class at conf=0.25 and at the F1-optimal conf
  - confusion matrix (classes + background), counts and column-normalised
  - PR curve, F1-confidence, P-confidence, R-confidence curves
  - model complexity: params, GFLOPs @640, FP32 CPU latency
  - sample prediction grid (GT green, predictions coloured by class)
All written to --out as PNGs + metrics.json + report.md.

Class ids follow the D-Fire / training convention: 0 = smoke, 1 = fire.

Run (WSL, CPU):
  python3 evaluate_model.py --ckpt <best.pth> --split val test --out results_v7
Single image:
  python3 evaluate_model.py --ckpt <best.pth> --image some.jpg --out results_v7
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

CLASSES = ("smoke", "fire")
COLORS = {0: (200, 200, 200), 1: (0, 90, 255)}  # BGR: smoke grey, fire orange-red
YOLOX_DIR = os.path.expanduser("~/wildfire_project/YOLOX")
DATA = os.path.expanduser("~/wildfire_project/dataset")
SPLITS = {
    "val": (f"{DATA}/extracted/data/val/images", f"{DATA}/extracted/data/val/labels"),
    "test": (f"{DATA}/data/test/images", f"{DATA}/extracted/data/test/labels"),
}
INPUT = 640
MAP_CONF, MAP_NMS = 0.001, 0.65   # same as the training exp's test_conf / nmsthre
OP_CONF = 0.25                    # conventional operating point for P/R/F1 + confusion

sys.path.insert(0, YOLOX_DIR)


# ---------------------------------------------------------------- model
def build_model(ckpt_path):
    from yolox.exp import get_exp
    exp = get_exp(exp_file=None, exp_name="yolox-s")
    exp.depth, exp.width, exp.act, exp.num_classes = 0.33, 0.50, "lrelu", len(CLASSES)
    model = exp.get_model()
    # Kaggle saved with NumPy 2.x (numpy._core); alias it for NumPy 1.x unpickling.
    if not hasattr(np, "_core"):
        import numpy.core as _nc
        import numpy.core.multiarray as _ncm
        sys.modules.setdefault("numpy._core", _nc)
        sys.modules.setdefault("numpy._core.multiarray", _ncm)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint/model mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    n_lcam = sum(1 for m in model.modules() if m.__class__.__name__ == "LCAM")
    n_hsig = sum(1 for m in model.modules() if isinstance(m, torch.nn.Hardsigmoid))
    print(f"loaded {os.path.basename(ckpt_path)}: strict match, LCAM x{n_lcam}, Hardsigmoid x{n_hsig}")
    return model.eval()


def preprocess(img):
    h, w = img.shape[:2]
    r = min(INPUT / h, INPUT / w)
    canvas = np.full((INPUT, INPUT, 3), 114, dtype=np.uint8)
    canvas[: int(h * r), : int(w * r)] = cv2.resize(img, (int(w * r), int(h * r)),
                                                    interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(canvas.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0)
    return x, r


@torch.no_grad()
def predict(model, img):
    from yolox.utils import postprocess
    x, r = preprocess(img)
    out = postprocess(model(x), len(CLASSES), MAP_CONF, MAP_NMS)[0]
    if out is None:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    out = out.numpy()
    return out[:, :4] / r, out[:, 4] * out[:, 5], out[:, 6].astype(int)


def load_gt(label_path, w, h):
    boxes, cls = [], []
    if os.path.exists(label_path):
        for line in open(label_path):
            p = line.split()
            if len(p) != 5:
                continue
            c, cx, cy, bw, bh = int(float(p[0])), *map(float, p[1:])
            if bw <= 0 or bh <= 0:
                continue
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
            cls.append(c)
    return np.array(boxes, dtype=np.float64).reshape(-1, 4), np.array(cls, dtype=int)


def box_iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


# ---------------------------------------------------------------- metrics
def coco_metrics(records, out_dir, split):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    images, anns, dets = [], [], []
    aid = 1
    for iid, r in enumerate(records, start=1):
        images.append({"id": iid, "file_name": r["name"], "width": r["w"], "height": r["h"]})
        for b, c in zip(r["gt_boxes"], r["gt_cls"]):
            anns.append({"id": aid, "image_id": iid, "category_id": int(c) + 1,
                         "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                         "area": float((b[2] - b[0]) * (b[3] - b[1])), "iscrowd": 0})
            aid += 1
        for b, s, c in zip(r["boxes"], r["scores"], r["cls"]):
            dets.append({"image_id": iid, "category_id": int(c) + 1,
                         "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                         "score": float(s)})
    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns,
                  "categories": [{"id": i + 1, "name": n} for i, n in enumerate(CLASSES)]}
    gt.createIndex()
    dt = gt.loadRes(dets) if dets else COCO()
    ev = COCOeval(gt, dt, "bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    names = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
             "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large"]
    overall = {n: float(v) for n, v in zip(names, ev.stats)}
    prec = ev.eval["precision"]  # [T, R, K, A, M]
    per_class = {}
    for k, name in enumerate(CLASSES):
        p_all = prec[:, :, k, 0, 2]
        p_50 = prec[0, :, k, 0, 2]
        per_class[name] = {
            "AP": float(np.mean(p_all[p_all > -1])) if (p_all > -1).any() else float("nan"),
            "AP50": float(np.mean(p_50[p_50 > -1])) if (p_50 > -1).any() else float("nan"),
            "n_gt": int(sum(1 for a in anns if a["category_id"] == k + 1)),
        }
    return overall, per_class


def pr_data(records, iou_thr=0.5):
    """Per-class (scores, tp) lists with greedy score-ordered matching at IoU>=iou_thr."""
    scores = {k: [] for k in range(len(CLASSES))}
    tps = {k: [] for k in range(len(CLASSES))}
    npos = {k: 0 for k in range(len(CLASSES))}
    for r in records:
        for k in range(len(CLASSES)):
            g = r["gt_boxes"][r["gt_cls"] == k]
            npos[k] += len(g)
            m = r["cls"] == k
            b, s = r["boxes"][m], r["scores"][m]
            order = np.argsort(-s)
            b, s = b[order], s[order]
            used = np.zeros(len(g), dtype=bool)
            ious = box_iou(b, g)
            for i in range(len(b)):
                tp = 0
                if len(g):
                    cand = np.where(~used)[0]
                    if len(cand):
                        j = cand[np.argmax(ious[i, cand])]
                        if ious[i, j] >= iou_thr:
                            used[j] = True
                            tp = 1
                scores[k].append(s[i]); tps[k].append(tp)
    return scores, tps, npos


def curves(scores, tps, npos):
    x = np.linspace(0, 1, 1000)
    res = {}
    for k in range(len(CLASSES)):
        s, t = np.array(scores[k]), np.array(tps[k])
        order = np.argsort(-s)
        s, t = s[order], t[order]
        ctp, cfp = np.cumsum(t), np.cumsum(1 - t)
        rec = ctp / max(npos[k], 1)
        prec = ctp / np.maximum(ctp + cfp, 1e-9)
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([1.0], prec, [0.0]))
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap50 = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
        r_at = np.interp(-x, -s, rec, left=0) if len(s) else np.zeros_like(x)
        p_at = np.interp(-x, -s, prec, left=1) if len(s) else np.ones_like(x)
        f1_at = 2 * p_at * r_at / (p_at + r_at + 1e-9)
        res[k] = {"rec": rec, "prec": prec, "mrec": mrec, "mpre": mpre, "ap50_voc": ap50,
                  "x": x, "p": p_at, "r": r_at, "f1": f1_at}
    return res


def prf_at(res, conf):
    i = int(np.clip(round(conf * 999), 0, 999))
    return {CLASSES[k]: {"precision": float(v["p"][i]), "recall": float(v["r"][i]),
                         "f1": float(v["f1"][i])} for k, v in res.items()}


def confusion(records, conf=OP_CONF, iou_thr=0.5):
    """Ultralytics-style: rows = predicted, cols = true; last index = background."""
    nc = len(CLASSES)
    mat = np.zeros((nc + 1, nc + 1), dtype=np.int64)
    for r in records:
        keep = r["scores"] >= conf
        db, dc = r["boxes"][keep], r["cls"][keep]
        gb, gc = r["gt_boxes"], r["gt_cls"]
        if len(gb) == 0:
            for c in dc:
                mat[c, nc] += 1
            continue
        if len(db) == 0:
            for c in gc:
                mat[nc, c] += 1
            continue
        ious = box_iou(gb, db)
        gi, di = np.where(ious > iou_thr)
        matched_g, matched_d = set(), set()
        if len(gi):
            order = np.argsort(-ious[gi, di])
            for g, d in zip(gi[order], di[order]):
                if g in matched_g or d in matched_d:
                    continue
                matched_g.add(g); matched_d.add(d)
                mat[dc[d], gc[g]] += 1
        for g, c in enumerate(gc):
            if g not in matched_g:
                mat[nc, c] += 1
        for d, c in enumerate(dc):
            if d not in matched_d:
                mat[c, nc] += 1
    return mat


# ---------------------------------------------------------------- plots
def plot_all(res, mat, out_dir, split):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    for k, v in res.items():
        ax.plot(v["mrec"], v["mpre"], lw=2, label=f"{CLASSES[k]} AP50={v['ap50_voc']:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02),
           title=f"Precision-Recall curve ({split}, IoU=0.5)")
    ax.grid(alpha=0.3); ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(f"{out_dir}/{split}_pr_curve.png", dpi=150); plt.close(fig)

    for key, label in (("f1", "F1"), ("p", "Precision"), ("r", "Recall")):
        fig, ax = plt.subplots(figsize=(7, 5))
        for k, v in res.items():
            ax.plot(v["x"], v[key], lw=1.5, label=CLASSES[k])
        mean = np.mean([v[key] for v in res.values()], 0)
        best = int(np.argmax(mean)) if key == "f1" else None
        lab = f"all classes {mean[best]:.3f} @ {res[0]['x'][best]:.3f}" if key == "f1" else "all classes"
        ax.plot(res[0]["x"], mean, lw=3, color="k", label=lab)
        ax.set(xlabel="Confidence", ylabel=label, xlim=(0, 1), ylim=(0, 1.02),
               title=f"{label}-Confidence curve ({split})")
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(f"{out_dir}/{split}_{key}_curve.png", dpi=150); plt.close(fig)

    labels = list(CLASSES) + ["background"]
    for norm in (False, True):
        m = mat.astype(float)
        if norm:
            m = m / np.maximum(m.sum(0, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(m, cmap="Blues")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if (i, j) == (len(CLASSES), len(CLASSES)):
                    continue
                txt = f"{m[i, j]:.2f}" if norm else f"{int(m[i, j])}"
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if m[i, j] > m.max() * 0.5 else "black", fontsize=11)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        ax.set(xlabel="True", ylabel="Predicted",
               title=f"Confusion matrix ({split}, conf={OP_CONF}, IoU=0.5)" + (" normalised" if norm else ""))
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/{split}_confusion_matrix{'_normalized' if norm else ''}.png", dpi=150)
        plt.close(fig)


def draw(img, boxes, scores, cls, gt_boxes=None, conf=OP_CONF):
    vis = img.copy()
    t = max(2, img.shape[1] // 300)
    if gt_boxes is not None:
        for b in gt_boxes:
            cv2.rectangle(vis, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (0, 200, 0), t)
    for b, s, c in zip(boxes, scores, cls):
        if s < conf:
            continue
        p1, p2 = tuple(map(int, b[:2])), tuple(map(int, b[2:]))
        cv2.rectangle(vis, p1, p2, COLORS[int(c)], t)
        cv2.putText(vis, f"{CLASSES[int(c)]} {s:.2f}", (p1[0], max(p1[1] - 6, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 + img.shape[1] / 2000, COLORS[int(c)], t)
    return vis


def sample_grid(records, img_dir, out_path, n=12):
    with_gt = [r for r in records if len(r["gt_boxes"])]
    if not with_gt:
        print("no labelled images in this subset; skipping sample grid")
        return
    picks = with_gt[:: max(1, len(with_gt) // n)][:n]
    tiles = []
    for r in picks:
        img = cv2.imread(os.path.join(img_dir, r["name"]))
        vis = draw(img, r["boxes"], r["scores"], r["cls"], r["gt_boxes"])
        vis = cv2.resize(vis, (400, 300))
        cv2.putText(vis, r["name"], (5, 292), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(vis)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [cv2.hconcat(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite(out_path, cv2.vconcat(rows))


# ---------------------------------------------------------------- complexity
def complexity(model, n=30):
    params = sum(p.numel() for p in model.parameters())
    gflops = None
    try:
        from thop import profile
        import copy
        macs, _ = profile(copy.deepcopy(model), inputs=(torch.zeros(1, 3, INPUT, INPUT),), verbose=False)
        gflops = 2 * macs / 1e9
    except Exception as e:
        print("GFLOPs unavailable:", e)
    x = torch.rand(1, 3, INPUT, INPUT) * 255
    with torch.no_grad():
        for _ in range(5):
            model(x)
        t0 = time.perf_counter()
        for _ in range(n):
            model(x)
    ms = (time.perf_counter() - t0) / n * 1e3
    return {"params_M": params / 1e6, "GFLOPs_640": gflops,
            "cpu_fp32_latency_ms": ms, "cpu_threads": torch.get_num_threads()}


# ---------------------------------------------------------------- main
def run_split(model, split, out_dir, limit):
    img_dir, lbl_dir = SPLITS[split]
    names = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg")))
    if limit:
        names = names[:limit]
    records = []
    t0 = time.time()
    for i, name in enumerate(names):
        img = cv2.imread(os.path.join(img_dir, name))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes, scores, cls = predict(model, img)
        gb, gc = load_gt(os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt"), w, h)
        records.append({"name": name, "w": w, "h": h, "boxes": boxes, "scores": scores,
                        "cls": cls, "gt_boxes": gb, "gt_cls": gc})
        if (i + 1) % 250 == 0:
            el = time.time() - t0
            print(f"  [{split}] {i + 1}/{len(names)}  {el / (i + 1) * 1e3:.0f} ms/img  "
                  f"eta {el / (i + 1) * (len(names) - i - 1) / 60:.1f} min", flush=True)

    n_gt = sum(len(r["gt_boxes"]) for r in records)
    n_labelled = sum(1 for r in records if len(r["gt_boxes"]))
    print(f"[{split}] {len(records)} images, {n_labelled} with objects, {n_gt} GT boxes")
    overall, per_class = coco_metrics(records, out_dir, split)
    scores, tps, npos = pr_data(records)
    res = curves(scores, tps, npos)
    mean_f1 = np.mean([v["f1"] for v in res.values()], 0)
    best_i = int(np.argmax(mean_f1))
    best_conf = float(res[0]["x"][best_i])
    mat = confusion(records)
    plot_all(res, mat, out_dir, split)
    sample_grid(records, img_dir, f"{out_dir}/{split}_sample_predictions.jpg")

    op = prf_at(res, OP_CONF)
    bf = prf_at(res, best_conf)
    summary = {
        "split": split, "images": len(records),
        "gt_boxes": {CLASSES[k]: npos[k] for k in npos},
        "coco": overall, "per_class": per_class,
        f"prf@conf{OP_CONF}": op,
        "prf@best_f1_conf": {"conf": best_conf, **bf},
        "confusion_matrix": {"rows_pred_cols_true": mat.tolist(),
                             "labels": list(CLASSES) + ["background"]},
    }
    for tag, d in (("op", op), ("best", bf)):
        summary[f"mean_{tag}"] = {m: float(np.mean([d[c][m] for c in CLASSES]))
                                  for m in ("precision", "recall", "f1")}
    return summary


def write_report(summaries, cx, ckpt, out_dir):
    L = [f"# Evaluation report — `{os.path.basename(ckpt)}` (FP32, PyTorch)\n",
         f"Classes: 0 = smoke, 1 = fire. mAP uses conf>{MAP_CONF}, NMS {MAP_NMS} (training exp "
         f"settings); P/R/F1 and confusion use conf={OP_CONF} unless noted, IoU 0.5.\n",
         "## Model complexity\n",
         "| Params (M) | GFLOPs @640 | CPU FP32 latency (ms) | CPU threads |",
         "|---|---|---|---|",
         f"| {cx['params_M']:.2f} | {cx['GFLOPs_640'] if cx['GFLOPs_640'] is None else round(cx['GFLOPs_640'], 2)} "
         f"| {cx['cpu_fp32_latency_ms']:.1f} | {cx['cpu_threads']} |\n",
         "CPU latency is a PC reference only — not the KV260/DPU figure.\n",
         "## Headline table\n",
         "| Split | Images | P | R | F1 | mAP@0.5 | mAP@0.75 | mAP@0.5:0.95 |",
         "|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        o, m = s["coco"], s["mean_op"]
        L.append(f"| {s['split']} | {s['images']} | {m['precision']:.3f} | {m['recall']:.3f} | "
                 f"{m['f1']:.3f} | {o['AP50']:.3f} | {o['AP75']:.3f} | {o['AP']:.3f} |")
    for s in summaries:
        o = s["coco"]
        L += [f"\n## {s['split']} split\n",
              "### Per-class\n",
              "| Class | GT boxes | AP@0.5 | AP@0.5:0.95 | P@0.25 | R@0.25 | F1@0.25 |",
              "|---|---|---|---|---|---|---|"]
        for c in CLASSES:
            pc, op = s["per_class"][c], s[f"prf@conf{OP_CONF}"][c]
            L.append(f"| {c} | {pc['n_gt']} | {pc['AP50']:.3f} | {pc['AP']:.3f} | "
                     f"{op['precision']:.3f} | {op['recall']:.3f} | {op['f1']:.3f} |")
        bf = s["prf@best_f1_conf"]
        L += [f"\nBest mean F1 = {s['mean_best']['f1']:.3f} at conf = {bf['conf']:.3f} "
              f"(P {s['mean_best']['precision']:.3f}, R {s['mean_best']['recall']:.3f}).\n",
              "### COCO breakdown\n",
              "| AP | AP50 | AP75 | AP_S | AP_M | AP_L | AR_1 | AR_10 | AR_100 |",
              "|---|---|---|---|---|---|---|---|---|",
              "| " + " | ".join(f"{o[k]:.3f}" for k in
                               ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
                                "AR_1", "AR_10", "AR_100")) + " |",
              "\n### Confusion matrix (rows = predicted, cols = true)\n",
              "| | " + " | ".join(s["confusion_matrix"]["labels"]) + " |",
              "|---" * (len(CLASSES) + 2) + "|"]
        for lab, row in zip(s["confusion_matrix"]["labels"], s["confusion_matrix"]["rows_pred_cols_true"]):
            L.append(f"| **{lab}** | " + " | ".join(str(v) for v in row) + " |")
        L.append(f"\nFigures: `{s['split']}_pr_curve.png`, `{s['split']}_f1_curve.png`, "
                 f"`{s['split']}_p_curve.png`, `{s['split']}_r_curve.png`, "
                 f"`{s['split']}_confusion_matrix*.png`, `{s['split']}_sample_predictions.jpg`")
    open(f"{out_dir}/report.md", "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", nargs="*", default=["val", "test"], choices=list(SPLITS))
    ap.add_argument("--image", help="single-image prediction instead of dataset evaluation")
    ap.add_argument("--conf", type=float, default=OP_CONF, help="display threshold for --image")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N images per split (smoke test)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(os.cpu_count())
    model = build_model(args.ckpt)

    if args.image:
        img = cv2.imread(args.image)
        t0 = time.perf_counter()
        boxes, scores, cls = predict(model, img)
        ms = (time.perf_counter() - t0) * 1e3
        out = os.path.join(args.out, "pred_" + os.path.basename(args.image))
        cv2.imwrite(out, draw(img, boxes, scores, cls, conf=args.conf))
        keep = scores >= args.conf
        print(f"{keep.sum()} detections >= {args.conf} in {ms:.0f} ms -> {out}")
        for b, s, c in zip(boxes[keep], scores[keep], cls[keep]):
            print(f"  {CLASSES[c]:5s} {s:.3f}  [{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}]")
        return

    cx = complexity(model)
    print("complexity:", cx)
    summaries = [run_split(model, s, args.out, args.limit) for s in args.split]
    json.dump({"ckpt": args.ckpt, "complexity": cx, "splits": summaries},
              open(f"{args.out}/metrics.json", "w"), indent=2)
    write_report(summaries, cx, args.ckpt, args.out)
    print(f"done -> {args.out}/report.md")


if __name__ == "__main__":
    main()
