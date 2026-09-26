#!/usr/bin/env python3
"""
lcam_video_4k.py  --  TIER 0: run the EXISTING wildfire LCAM-YOLOX model
                      (lcam_v5_2p5.xmodel) on 4K video.

No retraining, no recompiling, no changes to any existing file.
Reuses the fast video front-end developed for the COCO pipeline
(raw JPEG out of the AVI + optional scaled-DCT decode) and drives the
full LCAM graph through vitis_ai_library.GraphRunner, which dispatches
the 8 DPU subgraphs to the DPU and the 13 CPU subgraphs to the ARM cores.

The YOLOX conventions here are taken verbatim from the project's own
verified postprocess_detections.py / e2e_graphrunner.py:

  * input is RAW 0..255 **BGR** pixels -- NOT normalised by 255.
    int8 = pixel * 2^fix_point   (fix_point is -1 -> pixel/2 -> 0..127)
  * the model ALREADY applies sigmoid to obj/cls -- do not re-apply
  * 8400 anchors = 80x80 + 40x40 + 20x20 at strides 8/16/32
  * cxcy = (v + grid) * stride ;  wh = exp(v) * stride
  * classes: 0 = fire, 1 = smoke

  python3 lcam_video_4k.py <video.avi> [--reduce {1,2,4}] [--max-frames N]
                           [--conf C] [--save out.avi]
"""
import sys, os, time, struct, argparse
import numpy as np
import cv2

XMODEL = "/home/root/lcam_v5_2p5.xmodel"
NAMES = ["fire", "smoke"]
COLORS = [(0, 0, 255), (0, 165, 255)]          # fire = red, smoke = orange
REDUCE_FLAG = {1: cv2.IMREAD_COLOR,
               2: cv2.IMREAD_REDUCED_COLOR_2,
               4: cv2.IMREAD_REDUCED_COLOR_4}


# ----------------------------------------------------------------- video
def avi_jpeg_frames(path, limit=None):
    """Pull raw JPEG buffers out of an MJPEG AVI (no GStreamer)."""
    with open(path, "rb") as f:
        data = f.read()
    i = data.find(b"movi")
    if i < 0:
        raise RuntimeError("no movi list in %s" % path)
    p, n, out = i + 4, len(data), []
    while p + 8 <= n:
        cid = data[p:p + 4]
        sz = struct.unpack("<I", data[p + 4:p + 8])[0]
        p += 8
        if p + sz > n:
            break
        if cid[2:4] in (b"dc", b"db") and data[p:p + 2] == b"\xff\xd8":
            out.append(data[p:p + sz])
            if limit and len(out) >= limit:
                return out
        p += sz + (sz & 1)
    return out


def letterbox(img, size=640):
    oh, ow = img.shape[:2]
    s = min(size / ow, size / oh)
    nw, nh = int(ow * s), int(oh * s)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    cv2.resize(img, (nw, nh), dst=canvas[py:py + nh, px:px + nw],
               interpolation=cv2.INTER_LINEAR)
    return canvas, s, px, py


# ------------------------------------------------------------ yolox grid
def build_grid(order):
    gs, ss = [], []
    for (n, stride) in order:
        yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        g = np.stack((xv, yv), 2).reshape(-1, 2)
        gs.append(g)
        ss.append(np.full((g.shape[0], 1), stride, dtype=np.float32))
    return np.concatenate(gs, 0).astype(np.float32), np.concatenate(ss, 0)


def pick_order(pred):
    """Choose the concat order that yields in-image boxes (as the project's
    postprocess_detections.py does). Done ONCE, then reused."""
    best = None
    for label, order in (("80,40,20", [(80, 8), (40, 16), (20, 32)]),
                         ("20,40,80", [(20, 32), (40, 16), (80, 8)])):
        grids, strides = build_grid(order)
        if grids.shape[0] != pred.shape[0]:
            continue
        cxcy = (pred[:, 0:2] + grids) * strides
        wh = np.exp(np.clip(pred[:, 2:4], -10, 10)) * strides
        inside = np.mean((cxcy[:, 0] >= 0) & (cxcy[:, 0] <= 640) &
                         (cxcy[:, 1] >= 0) & (cxcy[:, 1] <= 640))
        sane = np.mean((wh[:, 0] > 1) & (wh[:, 0] < 1280) &
                       (wh[:, 1] > 1) & (wh[:, 1] < 1280))
        print("    order %-9s centres-in-image=%.3f sane-wh=%.3f" % (label, inside, sane))
        if best is None or inside + sane > best[0]:
            best = (inside + sane, label, grids, strides)
    return best[1], best[2], best[3]


def nms_np(boxes, scores, thr):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0., x2 - x1) * np.maximum(0., y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0., xx2 - xx1) * np.maximum(0., yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= thr]
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--xmodel", default=XMODEL)
    ap.add_argument("--reduce", type=int, default=1, choices=[1, 2, 4])
    ap.add_argument("--max-frames", type=int, default=40)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--nms", type=float, default=0.45)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    os.environ.setdefault(
        "XLNX_VART_FIRMWARE",
        "/lib/firmware/xilinx/kv260-benchmark-b4096/kv260-benchmark-b4096.xclbin")

    import xir
    from vitis_ai_library import GraphRunner

    print("=" * 70)
    print("TIER 0 -- existing wildfire LCAM-YOLOX on 4K video (no retraining)")
    print("=" * 70)
    print("xmodel : %s" % args.xmodel)

    bufs = avi_jpeg_frames(args.video, args.max_frames + args.warmup + 1)
    if not bufs:
        print("no JPEG frames recovered from %s" % args.video); return 1
    probe = cv2.imdecode(np.frombuffer(bufs[0], np.uint8), cv2.IMREAD_COLOR)
    SRC_H, SRC_W = probe.shape[:2]
    print("video  : %s  %dx%d  %d frames  (decode 1/%d)"
          % (args.video.split("/")[-1], SRC_W, SRC_H, len(bufs), args.reduce))

    g = xir.Graph.deserialize(args.xmodel)
    t0 = time.time()
    runner = GraphRunner.create_graph_runner(g)
    print("GraphRunner created in %.2f s" % (time.time() - t0))

    ins, outs = runner.get_inputs(), runner.get_outputs()
    in_arr = np.asarray(ins[0])
    it = ins[0].get_tensor()
    fp = None
    for k in ("fix_point", "fix_pos"):
        if it.has_attr(k):
            fp = it.get_attr(k); break
    size = in_arr.shape[1]
    print("input  : %s dtype=%s fix_point=%s   (RAW BGR 0..255, NOT /255)"
          % (list(in_arr.shape), in_arr.dtype, fp))
    for i, tb in enumerate(outs):
        print("output : [%d] %s" % (i, list(np.asarray(tb).shape)))

    flag = REDUCE_FLAG[args.reduce]
    writer = None
    grids = strides = None
    order_label = None

    t_dec = t_pre = t_inf = t_post = 0.0
    n = 0; ndet = 0
    per_class = {0: 0, 1: 0}
    t_wall0 = None

    for buf in bufs:
        if n >= args.max_frames + args.warmup:
            break
        a = time.time()
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), flag)
        b = time.time()
        if img is None:
            continue
        warm = n < args.warmup
        if not warm and t_wall0 is None:
            t_wall0 = a

        # NOTE: with --reduce N the decoded frame is SRC/N, not SRC. Boxes are
        # in THIS frame's coordinates, so clip and draw against it -- not
        # against the full-resolution probe size.
        CUR_H, CUR_W = img.shape[:2]

        canvas, scale, px, py = letterbox(img, size)
        data = canvas.astype(np.float32)              # RAW BGR pixels
        if fp is not None and np.issubdtype(in_arr.dtype, np.integer):
            data = np.clip(np.round(data * (2.0 ** fp)), -128, 127)
        in_arr[0] = data.astype(in_arr.dtype)
        c = time.time()

        jid = runner.execute_async(ins, outs)
        runner.wait(jid)
        d = time.time()

        pred = np.array(np.asarray(outs[0]))
        while pred.ndim > 2:
            pred = pred[0]
        if pred.shape[0] == 7 and pred.shape[1] == 8400:
            pred = pred.T

        if grids is None:
            print("\n  determining anchor concat order (once):")
            order_label, grids, strides = pick_order(pred)
            print("  using: %s" % order_label)
            ro, rc = pred[:, 4], pred[:, 5:]
            already = (ro.min() >= 0 and ro.max() <= 1 and
                       rc.min() >= 0 and rc.max() <= 1)
            print("  sigmoid already applied by model: %s\n" % already)

        cxcy = (pred[:, 0:2] + grids) * strides
        wh = np.exp(np.clip(pred[:, 2:4], -10, 10)) * strides
        obj, cls = pred[:, 4], pred[:, 5:]            # sigmoid ALREADY applied
        sc_all = obj[:, None] * cls
        cid = sc_all.argmax(1); csc = sc_all.max(1)
        keep = csc > args.conf

        boxes = np.zeros((0, 4)); scs = np.zeros(0); cids = np.zeros(0, dtype=int)
        if keep.any():
            bx = np.concatenate([cxcy[keep] - wh[keep] / 2,
                                 cxcy[keep] + wh[keep] / 2], 1)
            bx[:, [0, 2]] = np.clip((bx[:, [0, 2]] - px) / scale, 0, CUR_W)
            bx[:, [1, 3]] = np.clip((bx[:, [1, 3]] - py) / scale, 0, CUR_H)
            s_ = csc[keep]; c_ = cid[keep]
            fin = []
            for k in np.unique(c_):
                idx = np.where(c_ == k)[0]
                fin.extend(idx[nms_np(bx[idx], s_[idx], args.nms)])
            fin = np.array(sorted(fin, key=lambda i: -s_[i]), dtype=int)
            boxes, scs, cids = bx[fin], s_[fin], c_[fin]
        e = time.time()

        if args.save:
            if writer is None:
                writer = cv2.VideoWriter(args.save,
                                         cv2.VideoWriter_fourcc(*"MJPG"),
                                         10, (CUR_W, CUR_H))
                if not writer.isOpened():
                    print("WARNING: VideoWriter failed to open %s" % args.save)
            vis = img
            for i in range(len(boxes)):
                x1, y1, x2, y2 = [int(v) for v in boxes[i]]
                k = int(cids[i])
                col = COLORS[k % len(COLORS)]
                nm = NAMES[k] if k < len(NAMES) else "cls%d" % k
                cv2.rectangle(vis, (x1, y1), (x2, y2), col, 4)
                cv2.putText(vis, "%s %.2f" % (nm, scs[i]), (x1, max(y1 - 8, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 3)
            writer.write(vis)

        if not warm:
            t_dec += b - a; t_pre += c - b; t_inf += d - c; t_post += e - d
            ndet += len(boxes)
            for k in cids:
                per_class[int(k)] = per_class.get(int(k), 0) + 1
        n += 1

    if writer is not None:
        writer.release()

    m = n - args.warmup
    if m <= 0:
        print("no frames processed"); return 1
    wall = time.time() - t_wall0
    s = t_dec + t_pre + t_inf + t_post

    print("=" * 70)
    print("frames timed        : %d  (%d warm-up discarded)" % (m, args.warmup))
    print("detections          : %d total  (%.2f/frame)   fire=%d smoke=%d"
          % (ndet, ndet / m, per_class.get(0, 0), per_class.get(1, 0)))
    print("")
    print("--- mean ms per frame ---")
    print("  JPEG decode       : %8.2f   %5.1f%%" % (t_dec / m * 1e3, 100 * t_dec / s))
    print("  preprocess        : %8.2f   %5.1f%%" % (t_pre / m * 1e3, 100 * t_pre / s))
    print("  LCAM inference    : %8.2f   %5.1f%%   (8 DPU + 13 CPU subgraphs)"
          % (t_inf / m * 1e3, 100 * t_inf / s))
    print("  YOLOX decode+NMS  : %8.2f   %5.1f%%" % (t_post / m * 1e3, 100 * t_post / s))
    print("  ---------------------------------")
    print("  TOTAL             : %8.2f" % (s / m * 1e3))
    print("")
    print("END-TO-END          : %.2f FPS      wall-clock %.2f FPS"
          % (1000.0 / (s / m * 1e3), m / wall))
    if args.save:
        print("annotated video     : %s" % args.save)
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
