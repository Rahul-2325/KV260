#!/usr/bin/env python3
"""
Standalone YOLOv5-head detector for the Vitis-AI 2.5 `ofa_yolo_*` models
on the stock kv260-benchmark-b4096 DPU (fingerprint 0x101000016010407).

Self-contained: VART + numpy + cv2 only. Does not touch any existing file.

  python3 detect_ofa_yolo.py <xmodel> <image> [--bgr] [--conf 0.25] [--save out.jpg]
"""
import sys, time, argparse
import numpy as np
import cv2
import xir
import vart

COCO = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
    'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
    'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
    'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
    'remote','keyboard','cell phone','microwave','oven','toaster','sink',
    'refrigerator','book','clock','vase','scissors','teddy bear','hair drier',
    'toothbrush']

# YOLOv5 anchors, keyed by grid stride
ANCHORS = {
    8:  [(10, 13), (16, 30), (33, 23)],
    16: [(30, 61), (62, 45), (59, 119)],
    32: [(116, 90), (156, 198), (373, 326)],
}


def dpu_subgraph(graph):
    root = graph.get_root_subgraph()
    if root.is_leaf:
        return []
    return [s for s in root.toposort_child_subgraph()
            if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]


def letterbox(img, new=640, color=114):
    """Resize preserving aspect ratio, pad to square. Returns img, ratio, (dw, dh)."""
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new, new, 3), color, dtype=np.uint8)
    dw, dh = (new - nw) // 2, (new - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    return canvas, r, (dw, dh)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def nms(boxes, scores, thr):
    """boxes: (N,4) xyxy. Returns kept indices."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= thr]
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xmodel")
    ap.add_argument("image")
    ap.add_argument("--bgr", action="store_true",
                    help="feed BGR instead of RGB (default RGB)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--save", default=None)
    ap.add_argument("--runs", type=int, default=1, help="repeat for timing")
    args = ap.parse_args()

    g = xir.Graph.deserialize(args.xmodel)
    subs = dpu_subgraph(g)
    if len(subs) != 1:
        print("expected exactly 1 DPU subgraph, got %d" % len(subs))
        return 1
    runner = vart.Runner.create_runner(subs[0], "run")

    itensors = runner.get_input_tensors()
    otensors = runner.get_output_tensors()
    in_dims = tuple(itensors[0].dims)
    in_fp = itensors[0].get_attr("fix_point")
    in_scale = float(2 ** in_fp)
    size = in_dims[1]

    print("input  : %s fix_point=%d scale=%.1f" % (in_dims, in_fp, in_scale))
    out_meta = []
    for t in otensors:
        fp = t.get_attr("fix_point")
        d = tuple(t.dims)
        out_meta.append((d, fp, 2.0 ** -fp))
        print("output : %s fix_point=%d scale=%g  %s" % (d, fp, 2.0 ** -fp, t.name[:50]))

    img0 = cv2.imread(args.image)
    if img0 is None:
        print("cannot read image: %s" % args.image)
        return 1
    H0, W0 = img0.shape[:2]
    lb, ratio, (dw, dh) = letterbox(img0, size)
    feed = lb if args.bgr else cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)

    # float in [0,1] via scale 1/255, then quantise to int8
    q = np.round(feed.astype(np.float32) / 255.0 * in_scale)
    q = np.clip(q, -128, 127).astype(np.int8)

    indata = [np.ascontiguousarray(q.reshape(in_dims))]
    outdata = [np.empty(d, dtype=np.int8, order="C") for d, _, _ in out_meta]

    t0 = time.time()
    for _ in range(args.runs):
        jid = runner.execute_async(indata, outdata)
        runner.wait(jid)
    dt = (time.time() - t0) / args.runs
    print("\nDPU run: %.2f ms  (%.2f FPS)   [%d run(s)]" % (dt * 1e3, 1.0 / dt, args.runs))

    boxes, scores, classes = [], [], []
    for arr, (dims, fp, sc) in zip(outdata, out_meta):
        Gy, Gx, C = dims[1], dims[2], dims[3]
        stride = size // Gy
        if stride not in ANCHORS:
            print("unexpected stride %d for grid %d" % (stride, Gy))
            continue
        na = 3
        no = C // na                       # 85
        p = arr.astype(np.float32) * sc     # dequantise
        p = p.reshape(Gy, Gx, na, no)
        p = sigmoid(p)

        gy, gx = np.meshgrid(np.arange(Gy), np.arange(Gx), indexing="ij")
        gx = gx[..., None]
        gy = gy[..., None]

        cx = (p[..., 0] * 2.0 - 0.5 + gx) * stride
        cy = (p[..., 1] * 2.0 - 0.5 + gy) * stride
        aw = np.array([a[0] for a in ANCHORS[stride]], dtype=np.float32)
        ah = np.array([a[1] for a in ANCHORS[stride]], dtype=np.float32)
        bw = (p[..., 2] * 2.0) ** 2 * aw
        bh = (p[..., 3] * 2.0) ** 2 * ah

        obj = p[..., 4]
        cls = p[..., 5:]
        cls_id = cls.argmax(-1)
        cls_sc = cls.max(-1)
        conf = obj * cls_sc

        m = conf > args.conf
        if not m.any():
            continue
        bx, by = cx[m], cy[m]
        bwv, bhv = bw[m], bh[m]
        boxes.append(np.stack([bx - bwv / 2, by - bhv / 2,
                               bx + bwv / 2, by + bhv / 2], axis=1))
        scores.append(conf[m])
        classes.append(cls_id[m])

    if not boxes:
        print("\nNo detections above conf=%.2f" % args.conf)
        return 0

    boxes = np.concatenate(boxes, 0)
    scores = np.concatenate(scores, 0)
    classes = np.concatenate(classes, 0)

    # undo letterbox -> original image coords
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W0)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H0)

    print("\n%d candidates before NMS" % len(boxes))
    final = []
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        keep = nms(boxes[idx], scores[idx], args.iou)
        final.extend(idx[keep])
    final = sorted(final, key=lambda i: -scores[i])

    print("\n=== DETECTIONS (conf>=%.2f, iou=%.2f) ===" % (args.conf, args.iou))
    for i in final:
        c = int(classes[i])
        name = COCO[c] if c < len(COCO) else ("cls%d" % c)
        x1, y1, x2, y2 = boxes[i]
        print("  %-14s %.3f   [%4d,%4d,%4d,%4d]"
              % (name, scores[i], x1, y1, x2, y2))
    print("total: %d objects" % len(final))

    if args.save:
        vis = img0.copy()
        for i in final:
            c = int(classes[i])
            name = COCO[c] if c < len(COCO) else ("cls%d" % c)
            x1, y1, x2, y2 = [int(v) for v in boxes[i]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(vis, "%s %.2f" % (name, scores[i]), (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        cv2.imwrite(args.save, vis)
        print("saved: %s" % args.save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
