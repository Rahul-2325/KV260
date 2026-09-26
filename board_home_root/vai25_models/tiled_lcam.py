#!/usr/bin/env python3
"""
tiled_lcam.py -- SAHI-style tiled inference for LCAM-YOLOX on 4K frames.

THE PROBLEM THIS ADDRESSES
--------------------------
LCAM-YOLOX was trained on ~1280x720 images letterboxed to 640x640, i.e. the
content occupies 640x360 -- a 2x reduction. Feeding it a 3840x2160 frame
letterboxes to the SAME 640x360, which is a 6x reduction: every object is 3x
smaller than anything the model saw in training. Smoke plumes disappear.

Tiling restores training conditions: the 4K frame is cut into 1280x720 tiles,
each tile is letterboxed to 640x640 exactly as a training image would be, and
the per-tile boxes are mapped back to full-frame coordinates and merged with a
global per-class NMS.

Cost: one DPU+CPU pass per tile. At ~432 ms per pass a 3x3 grid is ~3.9 s per
frame -- an OFFLINE / high-accuracy mode, not real time. That trade is the
point of the experiment.

  python3 tiled_lcam.py <image.jpg> [--grid 3x3] [--overlap 0.0]
                        [--conf 0.30] [--out-prefix cmp]
"""
import sys, os, time, argparse
import numpy as np
import cv2

XMODEL = "/home/root/lcam_v5_2p5.xmodel"
NAMES = ["fire", "smoke"]
COLORS = [(0, 0, 255), (0, 165, 255)]


def letterbox(img, size=640):
    oh, ow = img.shape[:2]
    s = min(size / ow, size / oh)
    nw, nh = int(ow * s), int(oh * s)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    cv2.resize(img, (nw, nh), dst=canvas[py:py + nh, px:px + nw],
               interpolation=cv2.INTER_LINEAR)
    return canvas, s, px, py


def build_grid(order):
    gs, ss = [], []
    for (n, stride) in order:
        yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        g = np.stack((xv, yv), 2).reshape(-1, 2)
        gs.append(g)
        ss.append(np.full((g.shape[0], 1), stride, dtype=np.float32))
    return np.concatenate(gs, 0).astype(np.float32), np.concatenate(ss, 0)


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


class LCAM(object):
    """One GraphRunner, reused for every tile."""

    def __init__(self, xmodel):
        os.environ.setdefault(
            "XLNX_VART_FIRMWARE",
            "/lib/firmware/xilinx/kv260-benchmark-b4096/kv260-benchmark-b4096.xclbin")
        import xir
        from vitis_ai_library import GraphRunner
        # NOTE: the xir.Graph MUST outlive the GraphRunner. The runner holds a
        # non-owning pointer into it, so letting `g` fall out of scope at the
        # end of __init__ is a use-after-free and segfaults on the first
        # execute_async. Keep the reference on the instance.
        self.g = xir.Graph.deserialize(xmodel)
        self.r = GraphRunner.create_graph_runner(self.g)
        self.ins, self.outs = self.r.get_inputs(), self.r.get_outputs()
        self.ia = np.asarray(self.ins[0])
        t = self.ins[0].get_tensor()
        self.fp = t.get_attr("fix_point") if t.has_attr("fix_point") else 0
        self.size = self.ia.shape[1]
        self.grids = None
        self.n_calls = 0
        self.t_infer = 0.0

    def _order(self, pred):
        best = None
        for order in ([(80, 8), (40, 16), (20, 32)], [(20, 32), (40, 16), (80, 8)]):
            gr, st = build_grid(order)
            if gr.shape[0] != pred.shape[0]:
                continue
            cxcy = (pred[:, 0:2] + gr) * st
            wh = np.exp(np.clip(pred[:, 2:4], -10, 10)) * st
            sc = (np.mean((cxcy[:, 0] >= 0) & (cxcy[:, 0] <= 640) &
                          (cxcy[:, 1] >= 0) & (cxcy[:, 1] <= 640)) +
                  np.mean((wh[:, 0] > 1) & (wh[:, 0] < 1280)))
            if best is None or sc > best[0]:
                best = (sc, gr, st)
        return best[1], best[2]

    def detect(self, img, conf, nms_thr):
        """Returns boxes (xyxy, in img coords), scores, class ids."""
        H, W = img.shape[:2]
        canvas, scale, px, py = letterbox(img, self.size)
        d = np.clip(np.round(canvas.astype(np.float32) * (2.0 ** self.fp)), -128, 127)
        self.ia[0] = d.astype(self.ia.dtype)
        t0 = time.time()
        self.r.wait(self.r.execute_async(self.ins, self.outs))
        self.t_infer += time.time() - t0
        self.n_calls += 1

        pred = np.array(np.asarray(self.outs[0]))
        while pred.ndim > 2:
            pred = pred[0]
        if pred.shape[0] == 7 and pred.shape[1] == 8400:
            pred = pred.T
        if self.grids is None:
            self.grids, self.strides = self._order(pred)

        cxcy = (pred[:, 0:2] + self.grids) * self.strides
        wh = np.exp(np.clip(pred[:, 2:4], -10, 10)) * self.strides
        sc_all = pred[:, 4][:, None] * pred[:, 5:]     # sigmoid already applied
        cid = sc_all.argmax(1); csc = sc_all.max(1)
        keep = csc > conf
        if not keep.any():
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
        bx = np.concatenate([cxcy[keep] - wh[keep] / 2,
                             cxcy[keep] + wh[keep] / 2], 1)
        bx[:, [0, 2]] = np.clip((bx[:, [0, 2]] - px) / scale, 0, W)
        bx[:, [1, 3]] = np.clip((bx[:, [1, 3]] - py) / scale, 0, H)
        s_, c_ = csc[keep], cid[keep]
        fin = []
        for k in np.unique(c_):
            idx = np.where(c_ == k)[0]
            fin.extend(idx[nms_np(bx[idx], s_[idx], nms_thr)])
        fin = np.array(sorted(fin, key=lambda i: -s_[i]), dtype=int)
        return bx[fin], s_[fin], c_[fin]


def draw(img, boxes, scores, cids, thick=3, fs=1.0):
    vis = img.copy()
    for i in range(len(boxes)):
        x1, y1, x2, y2 = [int(v) for v in boxes[i]]
        k = int(cids[i]); col = COLORS[k % len(COLORS)]
        nm = NAMES[k] if k < len(NAMES) else "c%d" % k
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, thick)
        cv2.putText(vis, "%s %.2f" % (nm, scores[i]), (x1, max(y1 - 8, 24)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, col, thick)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--grid", default="3x3", help="tile grid, e.g. 3x3 or 4x3")
    ap.add_argument("--overlap", type=float, default=0.0, help="tile overlap fraction")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--nms", type=float, default=0.45)
    ap.add_argument("--out-prefix", default="cmp")
    ap.add_argument("--xmodel", default=XMODEL)
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print("cannot read %s" % args.image); return 1
    H, W = img.shape[:2]
    cols, rows = [int(v) for v in args.grid.lower().split("x")]

    print("=" * 72)
    print("TILED vs DOWNSCALED  --  LCAM-YOLOX on a %dx%d frame" % (W, H))
    print("=" * 72)

    net = LCAM(args.xmodel)

    # ---------------- A: whole frame downscaled (what we did before) -------
    t0 = time.time()
    b1, s1, c1 = net.detect(img, args.conf, args.nms)
    tA = time.time() - t0
    eff = min(640 / W, 640 / H)
    print("\n[A] DOWNSCALE WHOLE FRAME  (1 inference)")
    print("    %dx%d -> content %dx%d inside 640x640   (%.2fx reduction)"
          % (W, H, int(W * eff), int(H * eff), 1.0 / eff))
    print("    time %.0f ms   detections %d" % (tA * 1e3, len(b1)))
    for i in range(min(len(b1), 8)):
        print("      %-6s %.3f  [%d,%d,%d,%d]"
              % (NAMES[int(c1[i])], s1[i], *[int(v) for v in b1[i]]))

    # ---------------- B: tiled --------------------------------------------
    tw, th = W // cols, H // rows
    step_x = int(tw * (1 - args.overlap))
    step_y = int(th * (1 - args.overlap))
    origins = []
    y = 0
    while y + th <= H:
        x = 0
        while x + tw <= W:
            origins.append((x, y))
            if x + tw >= W:
                break
            x += step_x
        if y + th >= H:
            break
        y += step_y
    # make sure the right/bottom edges are covered
    for (ex, ey) in [(W - tw, y) for y in range(0, H - th + 1, step_y)] + \
                    [(x, H - th) for x in range(0, W - tw + 1, step_x)]:
        if (ex, ey) not in origins:
            origins.append((ex, ey))

    eff_t = min(640 / tw, 640 / th)
    print("\n[B] TILED  %s  overlap=%.0f%%  -> %d tiles of %dx%d"
          % (args.grid, args.overlap * 100, len(origins), tw, th))
    print("    each tile -> content %dx%d inside 640x640   (%.2fx reduction)"
          % (int(tw * eff_t), int(th * eff_t), 1.0 / eff_t))

    t0 = time.time()
    ab, asc, ac = [], [], []
    for (ox, oy) in origins:
        tile = img[oy:oy + th, ox:ox + tw]
        b, s, c = net.detect(tile, args.conf, args.nms)
        if len(b):
            b = b.copy()
            b[:, [0, 2]] += ox
            b[:, [1, 3]] += oy
            ab.append(b); asc.append(s); ac.append(c)
    tB = time.time() - t0

    if ab:
        B = np.concatenate(ab, 0); S = np.concatenate(asc, 0); C = np.concatenate(ac, 0)
        fin = []
        for k in np.unique(C):
            idx = np.where(C == k)[0]
            fin.extend(idx[nms_np(B[idx], S[idx], args.nms)])
        fin = np.array(sorted(fin, key=lambda i: -S[i]), dtype=int)
        B, S, C = B[fin], S[fin], C[fin]
    else:
        B = np.zeros((0, 4)); S = np.zeros(0); C = np.zeros(0, dtype=int)

    print("    time %.0f ms  (%.2f s)   detections %d  [after global NMS]"
          % (tB * 1e3, tB, len(B)))
    for i in range(min(len(B), 12)):
        print("      %-6s %.3f  [%d,%d,%d,%d]"
              % (NAMES[int(C[i])], S[i], *[int(v) for v in B[i]]))

    # ---------------- verdict ---------------------------------------------
    print("\n" + "=" * 72)
    print("RESULT")
    print("=" * 72)
    print("  downscaled : %2d detections   %6.0f ms   %5.2f FPS" % (len(b1), tA * 1e3, 1 / tA))
    print("  tiled      : %2d detections   %6.0f ms   %5.2f FPS" % (len(B), tB * 1e3, 1 / tB))
    if len(b1):
        print("  recall gain: %+.0f%%   cost: %.1fx slower"
              % (100.0 * (len(B) - len(b1)) / len(b1), tB / tA))
    else:
        print("  recall gain: %d detections found where downscaling found NONE" % len(B))
    fire_d = int((c1 == 0).sum()); smoke_d = int((c1 == 1).sum())
    fire_t = int((C == 0).sum()); smoke_t = int((C == 1).sum())
    print("  by class   : downscaled fire=%d smoke=%d | tiled fire=%d smoke=%d"
          % (fire_d, smoke_d, fire_t, smoke_t))

    o1 = "%s_downscaled.jpg" % args.out_prefix
    o2 = "%s_tiled.jpg" % args.out_prefix
    cv2.imwrite(o1, draw(img, b1, s1, c1, thick=6, fs=2.0))
    cv2.imwrite(o2, draw(img, B, S, C, thick=6, fs=2.0))
    print("  wrote %s and %s" % (o1, o2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
