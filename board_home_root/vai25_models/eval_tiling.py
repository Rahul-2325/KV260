#!/usr/bin/env python3
"""
eval_tiling.py -- RIGOROUS evaluation of tiled vs downscaled 4K inference,
against the project's own ground-truth labels.

"More boxes" is not "better". This measures recall / precision / F1 at
IoU >= 0.5 so the tiling gain can be stated as a real detection result.

METHOD
------
Synthetic-but-labelled 4K frames: N mosaics are built, each a 3x3 grid of
nine labelled valset images resized to exactly 1280x720. Because YOLO labels
are normalised, each image's boxes map into mosaic pixel coordinates exactly:

    x = (col + cx_norm) * 1280      w = w_norm * 1280
    y = (row + cy_norm) *  720      h = h_norm *  720

Both methods then see the IDENTICAL frame and the IDENTICAL ground truth,
so the comparison is fair. Aspect distortion from the resize is applied to
both equally.

  python3 eval_tiling.py [--mosaics 5] [--conf 0.30] [--grid 4x4] [--iou 0.5]
"""
import os, sys, glob, time, argparse
import numpy as np
import cv2

sys.path.insert(0, "/home/root/vai25_models")
from tiled_lcam import LCAM, nms_np, NAMES          # reuse the verified runner

CW, CH = 1280, 720
IMG_DIR = "/home/root/valset/images"
LBL_DIR = "/home/root/valset/labels"


def load_labelled(min_boxes=1):
    """valset images whose label file is non-empty."""
    out = []
    for ip in sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg"))):
        lp = os.path.join(LBL_DIR, os.path.basename(ip).replace(".jpg", ".txt"))
        if not os.path.exists(lp):
            continue
        rows = [r.split() for r in open(lp).read().strip().splitlines() if r.strip()]
        if len(rows) >= min_boxes:
            out.append((ip, [(int(r[0]), float(r[1]), float(r[2]),
                              float(r[3]), float(r[4])) for r in rows]))
    return out


def build_mosaic(items):
    """items: 9 x (path, labels). Returns 4K frame + GT boxes in frame coords."""
    canvas = np.zeros((CH * 3, CW * 3, 3), np.uint8)
    gt = []
    for i, (path, labs) in enumerate(items[:9]):
        im = cv2.imread(path)
        if im is None:
            continue
        r, c = divmod(i, 3)
        canvas[r * CH:(r + 1) * CH, c * CW:(c + 1) * CW] = cv2.resize(im, (CW, CH))
        for (cls, cx, cy, w, h) in labs:
            X = (c + cx) * CW
            Y = (r + cy) * CH
            W = w * CW
            H = h * CH
            gt.append([X - W / 2, Y - H / 2, X + W / 2, Y + H / 2, cls])
    return canvas, np.array(gt, dtype=np.float32).reshape(-1, 5)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def match(pred_b, pred_s, pred_c, gt, iou_thr, class_aware=True):
    """Greedy highest-score-first matching. Returns tp, fp, fn."""
    if len(gt) == 0:
        return 0, len(pred_b), 0
    if len(pred_b) == 0:
        return 0, 0, len(gt)
    order = np.argsort(-pred_s)
    pb, pc = pred_b[order], pred_c[order]
    M = iou_matrix(pb, gt[:, :4])
    used = np.zeros(len(gt), bool)
    tp = 0
    for i in range(len(pb)):
        best, bj = 0.0, -1
        for j in range(len(gt)):
            if used[j]:
                continue
            if class_aware and int(gt[j, 4]) != int(pc[i]):
                continue
            if M[i, j] > best:
                best, bj = M[i, j], j
        if bj >= 0 and best >= iou_thr:
            used[bj] = True
            tp += 1
    return tp, len(pb) - tp, len(gt) - tp


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def run_tiled(net, img, cols, rows, conf, nms_thr):
    H, W = img.shape[:2]
    tw, th = W // cols, H // rows
    ab, asc, ac = [], [], []
    for ry in range(rows):
        for cx in range(cols):
            ox, oy = cx * tw, ry * th
            b, s, c = net.detect(img[oy:oy + th, ox:ox + tw], conf, nms_thr)
            if len(b):
                b = b.copy(); b[:, [0, 2]] += ox; b[:, [1, 3]] += oy
                ab.append(b); asc.append(s); ac.append(c)
    if not ab:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    B = np.concatenate(ab); S = np.concatenate(asc); C = np.concatenate(ac)
    fin = []
    for k in np.unique(C):
        idx = np.where(C == k)[0]
        fin.extend(idx[nms_np(B[idx], S[idx], nms_thr)])
    fin = np.array(fin, int)
    return B[fin], S[fin], C[fin]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mosaics", type=int, default=5)
    ap.add_argument("--grid", default="4x4")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--nms", type=float, default=0.45)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--class-agnostic", action="store_true",
                    help="ignore class when matching (localisation only)")
    args = ap.parse_args()
    cols, rows = [int(v) for v in args.grid.lower().split("x")]

    lab = load_labelled()
    print("=" * 74)
    print("TILED vs DOWNSCALED -- measured against ground truth")
    print("=" * 74)
    print("labelled valset images (non-empty) : %d of 400" % len(lab))
    print("mosaics: %d  grid: %s  conf: %.2f  IoU: %.2f  class-aware: %s"
          % (args.mosaics, args.grid, args.conf, args.iou, not args.class_agnostic))

    net = LCAM("/home/root/lcam_v5_2p5.xmodel")

    agg = {"down": [0, 0, 0], "tile": [0, 0, 0]}
    t_down = t_tile = 0.0
    ngt = 0

    for m in range(args.mosaics):
        items = lab[m * 9:(m + 1) * 9]
        if len(items) < 9:
            break
        frame, gt = build_mosaic(items)
        ngt += len(gt)

        t0 = time.time()
        b1, s1, c1 = net.detect(frame, args.conf, args.nms)
        t_down += time.time() - t0

        t0 = time.time()
        b2, s2, c2 = run_tiled(net, frame, cols, rows, args.conf, args.nms)
        t_tile += time.time() - t0

        ca = not args.class_agnostic
        for key, (b, s, c) in (("down", (b1, s1, c1)), ("tile", (b2, s2, c2))):
            tp, fp, fn = match(b, s, c, gt, args.iou, ca)
            agg[key][0] += tp; agg[key][1] += fp; agg[key][2] += fn
        print("  mosaic %d: GT=%2d  downscaled=%2d  tiled=%2d"
              % (m + 1, len(gt), len(b1), len(b2)))

    n = args.mosaics
    print("")
    print("total ground-truth objects : %d  across %d 4K frames" % (ngt, n))
    print("")
    print("%-12s %5s %5s %5s   %9s %9s %9s   %10s" %
          ("method", "TP", "FP", "FN", "precision", "recall", "F1", "ms/frame"))
    print("-" * 74)
    for key, label, tt in (("down", "downscaled", t_down), ("tile", "tiled " + args.grid, t_tile)):
        tp, fp, fn = agg[key]
        p, r, f = prf(tp, fp, fn)
        print("%-12s %5d %5d %5d   %9.3f %9.3f %9.3f   %10.0f"
              % (label, tp, fp, fn, p, r, f, tt / n * 1e3))
    print("-" * 74)
    dp, dr, df = prf(*agg["down"])
    tp_, tr, tf = prf(*agg["tile"])
    print("delta        recall %+.3f (%+.0f%%)   precision %+.3f   F1 %+.3f   cost %.1fx"
          % (tr - dr, 100 * (tr - dr) / dr if dr else float('nan'),
             tp_ - dp, tf - df, t_tile / t_down if t_down else 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
