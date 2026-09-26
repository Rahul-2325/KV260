#!/usr/bin/env python3
"""
eval_map.py -- run ON THE BOARD.

Detection accuracy of the hybrid pipeline over a labelled validation set,
so the accuracy claim rests on a dataset rather than on one demo image.

Runs the same pipeline as hybrid_pipeline.py (DPU subgraphs on the DPU, the
four LCAM gates on the custom accelerator, glue in numpy), decodes with the
same logic as postprocess_detections.py, and scores against YOLO-format
ground truth.

  --gates ip     accelerator    <- the configuration being reported
  --gates numpy  software gates <- control; any AP difference is the IP's

Metric: AP per class at IoU 0.5, all-point interpolation (VOC2010 style),
plus mAP@[.5:.05:.95] for a COCO-comparable figure. Greedy matching, each
ground-truth box claimed at most once, detections consumed in confidence
order -- the standard protocol.

Ground truth is YOLO format: `cls cx cy w h`, normalised to the ORIGINAL
image, so it is converted to pixel xyxy using that image's own dimensions.
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
import vart
import xir

sys.path.insert(0, "/home/root")
sys.path.insert(0, "/home/root/hybrid_pkg")
from hybrid_pipeline import (LcamCU, dpu_round, run_cpu_subgraph, build_lut,
                             build_int8_path, sg_out, is_gate, gate_params,
                             letterbox, XMODEL)
from postprocess_detections import build_grid, decode


def load_gt(path, ow, oh):
    """YOLO `cls cx cy w h` (normalised) -> [(cls, x1,y1,x2,y2), ...] pixels."""
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path):
        p = line.split()
        if len(p) < 5:
            continue
        c = int(float(p[0]))
        cx, cy, w, h = [float(v) for v in p[1:5]]
        out.append((c, (cx - w / 2) * ow, (cy - h / 2) * oh,
                    (cx + w / 2) * ow, (cy + h / 2) * oh))
    return out


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = ((a[2] - a[0]) * (a[3] - a[1]) +
          (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def average_precision(dets, gts, cls, thr):
    """dets: (img, cls, score, box). gts: {img: [(cls,box)...]}. All-point AP."""
    d = sorted([x for x in dets if x[1] == cls], key=lambda x: -x[2])
    npos = sum(1 for g in gts.values() for c, *_ in g if c == cls)
    if npos == 0:
        return None, 0
    claimed = {}
    tp, fp = np.zeros(len(d)), np.zeros(len(d))
    for i, (img, _, _, box) in enumerate(d):
        best, bi = 0.0, -1
        for j, (c, *gb) in enumerate(gts.get(img, [])):
            if c != cls or claimed.get((img, j)):
                continue
            v = iou(box, gb)
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
    # all-point interpolation
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for k in range(len(mpre) - 2, -1, -1):
        mpre[k] = max(mpre[k], mpre[k + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])), npos


def confusion_matrix(dets, gts, thr=0.5):
    """
    Is a wrong-class detection a CLASSIFICATION mistake (real object, wrong
    label) or a LOCALISATION mistake (nothing there at all)? average_precision()
    can't tell you: it only matches a detection against ground-truth boxes of
    its OWN predicted class, so a "fire" box sitting on top of a real "smoke"
    object simply looks like one false positive for fire and one missed
    detection for smoke -- the connection between the two is invisible in
    the AP table.

    This matches every detection against the best-IoU ground-truth box in
    its image REGARDLESS of class (same greedy, confidence-ordered, claim-
    once protocol as average_precision, just not filtered by class first).

    Returns:
      matrix[(pred_cls, gt_cls)] = count of detections at IoU>=thr
      no_match_imgs = [(img, pred_cls, score)] for detections with NO
                      ground-truth box at IoU>=thr anywhere in the image --
                      genuine hallucinations, not class confusion
    """
    from collections import defaultdict
    d = sorted(dets, key=lambda x: -x[2])
    claimed = set()
    matrix = defaultdict(int)
    examples = defaultdict(list)
    no_match_imgs = []
    for img, pcls, score, box in d:
        best, bi, bcls = 0.0, -1, None
        for j, (c, *gb) in enumerate(gts.get(img, [])):
            if (img, j) in claimed:
                continue
            v = iou(box, gb)
            if v > best:
                best, bi, bcls = v, j, c
        if best >= thr and bi >= 0:
            claimed.add((img, bi))
            matrix[(pcls, bcls)] += 1
            if len(examples[(pcls, bcls)]) < 4:
                examples[(pcls, bcls)].append((img, score, best))
        else:
            no_match_imgs.append((img, pcls, score))
    return matrix, no_match_imgs, examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="/home/root/valset")
    ap.add_argument("--gates", choices=("ip", "numpy"), default="ip")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--nms", type=float, default=0.45)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--classes", default="fire,smoke")
    ap.add_argument("--confusion", action="store_true",
                    help="also print a class-confusion breakdown: of the "
                         "wrong-class detections, how many actually sit on "
                         "a real object of the OTHER class (true confusion) "
                         "vs no real object at all (hallucination)")
    args = ap.parse_args()
    names = args.classes.split(",")

    g = xir.Graph.deserialize(XMODEL)
    subs = g.get_root_subgraph().toposort_child_subgraph()
    runners, kinds, luts = {}, {}, {}
    for i, s in enumerate(subs):
        kinds[i] = s.get_attr("device").upper() if s.has_attr("device") else "USER"
        if kinds[i] == "DPU":
            runners[i] = vart.Runner.create_runner(s, "run")
        elif kinds[i] == "CPU" and not is_gate(s):
            r = build_lut(s)
            if r is not None:
                luts[i] = ("lut",) + r
            else:
                r = build_int8_path(s)
                if r is not None:
                    luts[i] = ("int8",) + r

    cu = LcamCU() if args.gates == "ip" else None
    in_name = sg_out(subs[0])[0].name
    in_fp = sg_out(subs[0])[0].get_attr("fix_point")
    grids, strides = build_grid([(80, 8), (40, 16), (20, 32)])

    imgs = sorted(glob.glob(os.path.join(args.set, "images", "*")))
    if args.limit:
        imgs = imgs[:args.limit]
    print("evaluating %d images, gates=%s\n" % (len(imgs), args.gates))

    dets, gts, lat = [], {}, []
    t_start = time.perf_counter()

    for n, ipath in enumerate(imgs):
        img = cv2.imread(ipath)
        if img is None:
            continue
        key = os.path.splitext(os.path.basename(ipath))[0]
        oh, ow = img.shape[:2]
        gts[key] = load_gt(os.path.join(args.set, "labels", key + ".txt"), ow, oh)

        canvas, scale, px, py, _, _ = letterbox(img)
        quant = np.clip(np.round(canvas.astype(np.float32) * (2.0 ** in_fp)),
                        -128, 127).astype(np.int8).reshape(1, 640, 640, 3)

        t0 = time.perf_counter()
        vals = {in_name: quant}
        for i, s in enumerate(subs):
            if kinds[i] == "USER":
                continue
            if kinds[i] == "DPU":
                r = runners[i]
                its, ots = r.get_input_tensors(), r.get_output_tensors()
                ins = [np.ascontiguousarray(vals[t.name]) for t in its]
                outs = [np.zeros(tuple(t.dims), dtype=np.int8) for t in ots]
                r.wait(r.execute_async(ins, outs))
                for t, o in zip(ots, outs):
                    vals[t.name] = o
            elif cu is not None and is_gate(s):
                ft, wt, fpf, fpw, fpo = gate_params(s)
                vals[sg_out(s)[0].name] = cu.gate(vals[ft], vals[wt], fpf, fpw, fpo)
            elif i in luts:
                kind, src, dst, f = luts[i]
                vals[dst] = (f[vals[src].view(np.uint8)]
                             if kind == "lut" else f(vals[src]))
            else:
                run_cpu_subgraph(s, vals)
        lat.append((time.perf_counter() - t0) * 1e3)

        pred = np.asarray(vals[sg_out(subs[-1])[0].name], dtype=np.float32)
        while pred.ndim > 2:
            pred = pred[0]
        if pred.shape[0] == 7:
            pred = pred.T

        cxcy, wh = decode(pred, grids, strides)
        obj, cls = pred[:, 4], pred[:, 5:]          # already sigmoided
        sc = obj[:, None] * cls
        cid, csc = sc.argmax(1), sc.max(1)
        keep = csc > args.conf
        if keep.sum():
            b = np.concatenate([cxcy[keep] - wh[keep] / 2,
                                cxcy[keep] + wh[keep] / 2], 1)
            b[:, [0, 2]] = np.clip((b[:, [0, 2]] - px) / scale, 0, ow)
            b[:, [1, 3]] = np.clip((b[:, [1, 3]] - py) / scale, 0, oh)
            s_, c_ = csc[keep], cid[keep]
            for k in np.unique(c_):
                m = c_ == k
                bb, ss = b[m], s_[m]
                idx = cv2.dnn.NMSBoxes(
                    [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                     for x1, y1, x2, y2 in bb],
                    ss.astype(float).tolist(), args.conf, args.nms)
                for q in np.array(idx).flatten() if len(idx) else []:
                    dets.append((key, int(k), float(ss[q]), tuple(bb[q])))

        if (n + 1) % 50 == 0:
            print("  %d/%d  (%.1f ms/frame)" % (n + 1, len(imgs), np.mean(lat)))

    wall = time.perf_counter() - t_start
    print("\n%d images in %.1f s   mean %.2f ms/frame   %.2f FPS"
          % (len(lat), wall, np.mean(lat), 1000.0 / np.mean(lat)))
    print("detections kept: %d   ground-truth objects: %d\n"
          % (len(dets), sum(len(v) for v in gts.values())))

    print("%-8s %10s %10s %10s" % ("class", "objects", "AP@0.5", "AP@.5:.95"))
    print("-" * 42)
    m50, m5095, ncls = [], [], 0
    for c, nm in enumerate(names):
        ap50, npos = average_precision(dets, gts, c, 0.5)
        if ap50 is None:
            print("%-8s %10s %10s %10s" % (nm, 0, "-", "-"))
            continue
        aps = [average_precision(dets, gts, c, t)[0]
               for t in np.arange(0.5, 0.96, 0.05)]
        aps = [a for a in aps if a is not None]
        a95 = float(np.mean(aps)) if aps else 0.0
        m50.append(ap50); m5095.append(a95); ncls += 1
        print("%-8s %10d %10.4f %10.4f" % (nm, npos, ap50, a95))
    if ncls:
        print("-" * 42)
        print("%-8s %10s %10.4f %10.4f"
              % ("mAP", "", float(np.mean(m50)), float(np.mean(m5095))))

    if args.confusion:
        matrix, no_match, examples = confusion_matrix(dets, gts, 0.5)
        print()
        print("=" * 62)
        print("CLASS CONFUSION  (at IoU>=0.5, conf>%.2f, greedy claim-once)" % args.conf)
        print("=" * 62)
        print("%-22s %10s" % ("predicted -> actual", "count"))
        print("-" * 62)
        total_correct = total_confused = 0
        for pc, pname in enumerate(names):
            for gc, gname in enumerate(names):
                n = matrix.get((pc, gc), 0)
                if n == 0:
                    continue
                tag = "CORRECT" if pc == gc else "*** CONFUSED ***"
                print("%-22s %10d   %s" % ("%s -> %s" % (pname, gname), n, tag))
                if pc == gc:
                    total_correct += n
                else:
                    total_confused += n
        print("-" * 62)
        print("correctly classified (localised right, right label) : %d" % total_correct)
        print("CLASS CONFUSED (localised right, WRONG label)        : %d" % total_confused)
        print("no matching ground truth at all (hallucination)      : %d" % len(no_match))
        if total_correct + total_confused:
            rate = 100.0 * total_confused / (total_correct + total_confused)
            print()
            print("of every detection that landed on a REAL object, %.1f%% had the wrong label"
                  % rate)
        print()
        for (pc, gc), exs in examples.items():
            if pc == gc:
                continue
            print("  example '%s predicted, actually %s' cases (image, score, IoU):"
                  % (names[pc], names[gc]))
            for img, score, iouv in exs:
                print("    %-24s conf=%.3f  IoU=%.3f" % (img, score, iouv))

    if cu:
        cu.close()


if __name__ == "__main__":
    main()
