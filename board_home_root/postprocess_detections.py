#!/usr/bin/env python3
"""
postprocess_detections.py
Decodes the [1,8400,7] YOLOX head output into bounding boxes, runs NMS,
prints detections and writes an annotated image.

YOLOX head layout per anchor: [cx, cy, w, h, obj, cls0, cls1]
  cx,cy are grid-relative offsets  -> (v + grid) * stride
  w,h   are log-space              -> exp(v) * stride
8400 anchors = 80x80(6400) + 40x40(1600) + 20x20(400) at strides 8/16/32.
The concat order is verified empirically (both orders are tried and the
one producing in-image boxes is used).

TWO WAYS TO USE THIS FILE:
  1. CLI, decoding a previously saved .npz -- unchanged, see main() below.
     Used when the raw detection tensor was captured separately (e.g.
     hybrid_pipeline.py --save-npz) and you want to re-run decoding/NMS
     without re-running inference on the board.
  2. Imported: `from postprocess_detections import run_postprocess` and
     call it directly with the tensor already in memory. This is what lets
     hybrid_pipeline.py's `--out` flag produce an annotated image in ONE
     command instead of two -- see the timing note in run_postprocess().
"""
import argparse
import time

import numpy as np


def build_grid(order):
    """Return (grids, strides) arrays of shape (8400,2) and (8400,1)."""
    gs, ss = [], []
    for (n, stride) in order:
        yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        g = np.stack((xv, yv), 2).reshape(-1, 2)
        gs.append(g)
        ss.append(np.full((g.shape[0], 1), stride, dtype=np.float32))
    return np.concatenate(gs, 0).astype(np.float32), np.concatenate(ss, 0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def decode(pred, grids, strides):
    cxcy = (pred[:, 0:2] + grids) * strides
    wh = np.exp(np.clip(pred[:, 2:4], -10, 10)) * strides
    return cxcy, wh


def run_postprocess(pred, scale, px, py, ow, oh, image, out,
                    conf=0.30, nms=0.45, classes='fire,smoke', verbose=True):
    """
    Decode + NMS + draw + save, on an ALREADY-IN-MEMORY prediction tensor.

    This is the guts of the old CLI main(), factored out so it can be
    called straight after hybrid_pipeline.py's inference loop -- no .npz
    round-trip needed for the common case of "run once, see the picture".

    Returns (detections, timings_ms) where timings_ms = {decode, nms_draw,
    imwrite, total} -- printed by the caller (or here, if verbose) so the
    cost of this step is always visible and never silently folded into a
    hardware throughput number.
    """
    import cv2
    names = classes.split(',')
    t_all0 = time.perf_counter()

    pred = np.asarray(pred, dtype=np.float32)
    while pred.ndim > 2:
        pred = pred[0]
    if pred.shape[0] == 7 and pred.shape[1] == 8400:
        pred = pred.T
    n_anchor, n_feat = pred.shape

    t0 = time.perf_counter()
    best = None
    for label, order in (("80,40,20", [(80, 8), (40, 16), (20, 32)]),
                         ("20,40,80", [(20, 32), (40, 16), (80, 8)])):
        grids, strides = build_grid(order)
        if grids.shape[0] != n_anchor:
            continue
        cxcy, wh = decode(pred, grids, strides)
        inside = np.mean((cxcy[:, 0] >= 0) & (cxcy[:, 0] <= 640) &
                         (cxcy[:, 1] >= 0) & (cxcy[:, 1] <= 640))
        sane_wh = np.mean((wh[:, 0] > 1) & (wh[:, 0] < 1280) &
                          (wh[:, 1] > 1) & (wh[:, 1] < 1280))
        score = inside + sane_wh
        if best is None or score > best[0]:
            best = (score, label, cxcy, wh)
    _, label, cxcy, wh = best

    # The xmodel ALREADY applies sigmoid to obj/cls -- those are the 6
    # `sigmoid` ops in CPU subgraphs 10-17 (see PROJECT_HISTORY §29.4).
    # Applying it a second time squashes every score to ~0.30 and yields
    # hundreds of identical-confidence false boxes. Detect and skip.
    raw_obj, raw_cls = pred[:, 4], pred[:, 5:]
    already = (raw_obj.min() >= 0.0 and raw_obj.max() <= 1.0 and
               raw_cls.min() >= 0.0 and raw_cls.max() <= 1.0)
    obj, cls = (raw_obj, raw_cls) if already else (sigmoid(raw_obj), sigmoid(raw_cls))
    scores = obj[:, None] * cls
    cls_id = scores.argmax(1)
    cls_sc = scores.max(1)
    keep = cls_sc > conf
    t_decode = (time.perf_counter() - t0) * 1e3

    dets = []
    t0 = time.perf_counter()
    img = cv2.imread(image) if image else None
    if keep.sum() > 0:
        b = np.concatenate([cxcy[keep] - wh[keep] / 2, cxcy[keep] + wh[keep] / 2], 1)
        s = cls_sc[keep]; c = cls_id[keep]
        b[:, [0, 2]] = np.clip((b[:, [0, 2]] - px) / scale, 0, ow)
        b[:, [1, 3]] = np.clip((b[:, [1, 3]] - py) / scale, 0, oh)

        colors = [(0, 0, 255), (0, 165, 255)]
        for k in np.unique(c):
            m = c == k
            boxes = b[m]; sc = s[m]
            idx = cv2.dnn.NMSBoxes(
                [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                 for x1, y1, x2, y2 in boxes],
                sc.astype(float).tolist(), conf, nms)
            for i in (np.array(idx).flatten() if len(idx) else []):
                x1, y1, x2, y2 = boxes[i].astype(int)
                nm = names[k] if k < len(names) else "cls%d" % k
                dets.append((nm, float(sc[i]), (int(x1), int(y1), int(x2), int(y2))))
                if img is not None:
                    col = colors[k % len(colors)]
                    cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
                    cv2.putText(img, "%s %.2f" % (nm, sc[i]), (x1, max(y1 - 5, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    t_nms_draw = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    if out and img is not None:
        cv2.imwrite(out, img)
    t_imwrite = (time.perf_counter() - t0) * 1e3

    timings = {"decode": t_decode, "nms_draw": t_nms_draw, "imwrite": t_imwrite,
              "total": (time.perf_counter() - t_all0) * 1e3}

    if verbose:
        print("  concat order: %s   anchors above conf=%.2f: %d"
              % (label, conf, int(keep.sum())))
        for nm, sc, box in dets:
            print("  %-6s conf=%.3f  box=%s" % (nm, sc, list(box)))
        print("  TOTAL DETECTIONS: %d" % len(dets))
        if out and img is not None:
            print("  annotated image written to %s" % out)
        # THIS is the real answer to "does splitting into two commands cost
        # FPS": here is every millisecond postprocessing actually costs,
        # measured, not assumed. It is deliberately kept OUT of the
        # hardware pipeline's reported FPS -- see hybrid_pipeline.py --out.
        print("  postprocess cost: decode %.2f + nms/draw %.2f + imwrite %.2f"
              " = %.2f ms  (NOT counted in the hardware FPS above)"
              % (t_decode, t_nms_draw, t_imwrite, timings["total"]))

    return dets, timings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='/home/root/e2e_outputs.npz')
    ap.add_argument('--image', default='/home/root/WEB09971.jpg')
    ap.add_argument('--out', default='/home/root/result.jpg')
    ap.add_argument('--conf', type=float, default=0.30)
    ap.add_argument('--nms', type=float, default=0.45)
    ap.add_argument('--classes', default='fire,smoke')
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    pred = z['out_0']
    print("raw output shape: %s  dtype=%s" % (list(np.asarray(pred).shape), pred.dtype))
    scale = float(z['scale']); px, py = [int(v) for v in z['pad']]
    ow, oh = [int(v) for v in z['orig']]
    print("letterbox: scale=%.4f pad=(%d,%d) orig=%dx%d" % (scale, px, py, ow, oh))
    if 'latency_ms' in z:
        lat = z['latency_ms']
        print("measured latency: %.2f ms  -> %.2f FPS" % (lat.mean(), 1000 / lat.mean()))

    run_postprocess(pred, scale, px, py, ow, oh, args.image, args.out,
                    args.conf, args.nms, args.classes, verbose=True)


if __name__ == '__main__':
    main()
