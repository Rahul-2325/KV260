#!/usr/bin/env python3
"""
OPTIMISED end-to-end video benchmark (same maths, same results as
bench_video.py -- only faster).

Two changes, both on the ARM side:

 1. PREPROCESS: the float multiply + round over 1.23M elements is replaced
    by a precomputed 256-entry int8 lookup table, and the BGR->RGB swap is
    folded into the same indexing operation.

 2. POSTPROCESS: instead of running sigmoid over all 2.14M output elements,
    the objectness channel is thresholded while still in the RAW int8
    domain (sigmoid is monotonic, so conf > T implies sigmoid(obj) > T,
    i.e. obj_raw > logit(T)/scale). Only the few surviving anchors are
    dequantised and decoded.

  python3 bench_video_opt.py <xmodel> <video> [--max-frames N] [--conf C] [--save out.avi]
"""
import sys, time, argparse, math
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

ANCHORS = {8:  [(10,13),(16,30),(33,23)],
           16: [(30,61),(62,45),(59,119)],
           32: [(116,90),(156,198),(373,326)]}


def dpu_subgraph(graph):
    return [s for s in graph.get_root_subgraph().toposort_child_subgraph()
            if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def nms(boxes, scores, thr):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = np.maximum(0., x2-x1) * np.maximum(0., y2-y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0., xx2-xx1) * np.maximum(0., yy2-yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= thr]
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xmodel"); ap.add_argument("video")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--max-frames", type=int, default=100)
    ap.add_argument("--save", default=None)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    g = xir.Graph.deserialize(args.xmodel)
    runner = vart.Runner.create_runner(dpu_subgraph(g)[0], "run")
    it = runner.get_input_tensors(); ot = runner.get_output_tensors()
    in_dims = tuple(it[0].dims)
    in_scale = float(2 ** it[0].get_attr("fix_point"))
    size = in_dims[1]
    out_meta = [(tuple(t.dims), 2.0 ** -t.get_attr("fix_point")) for t in ot]
    outdata = [np.empty(d, dtype=np.int8, order="C") for d,_ in out_meta]

    # --- OPT 1: int8 quantisation lookup table (index by uint8 pixel) ---
    LUT = np.clip(np.round(np.arange(256, dtype=np.float32) / 255.0 * in_scale),
                  -128, 127).astype(np.int8)

    # --- OPT 2: precompute per-head constants + raw objectness thresholds ---
    logit_t = math.log(args.conf / (1.0 - args.conf))
    heads = []
    for (dims, sc) in out_meta:
        Gy, Gx, C = dims[1], dims[2], dims[3]
        stride = size // Gy
        anc = np.array(ANCHORS[stride], dtype=np.float32)
        heads.append({"Gy":Gy, "Gx":Gx, "no":C//3, "stride":stride,
                      "sc":sc, "anc":anc, "thr_raw": logit_t / sc})

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("cannot open video: %s" % args.video); return 1
    VW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    VH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("model : %s   [OPTIMISED PIPELINE]" % args.xmodel.split("/")[-1])
    print("video : %s  %dx%d" % (args.video.split("/")[-1], VW, VH))

    # letterbox geometry is constant for a fixed-size video -> compute once
    r = min(size / VH, size / VW)
    nh, nw = int(round(VH * r)), int(round(VW * r))
    dw, dh = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)

    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"MJPG"),
                                 30, (VW, VH))

    t_dec = t_pre = t_dpu = t_post = 0.0
    n = 0; ndet = 0; t_wall0 = None
    while n < args.max_frames + args.warmup:
        a = time.time()
        ok, frame = cap.read()
        b = time.time()
        if not ok:
            break
        warm = n < args.warmup
        if not warm and t_wall0 is None:
            t_wall0 = a

        cv2.resize(frame, (nw, nh), dst=canvas[dh:dh+nh, dw:dw+nw],
                   interpolation=cv2.INTER_LINEAR)
        q = LUT[canvas[:, :, ::-1]]                     # BGR->RGB + quantise
        indata = [np.ascontiguousarray(q.reshape(in_dims))]
        c = time.time()

        jid = runner.execute_async(indata, outdata)
        runner.wait(jid)
        d = time.time()

        boxes, scores, classes = [], [], []
        for arr, h in zip(outdata, heads):
            av = arr.reshape(h["Gy"], h["Gx"], 3, h["no"])
            m = av[..., 4] > h["thr_raw"]               # RAW-domain gate
            if not m.any():
                continue
            gy, gx, ai = np.nonzero(m)
            p = sigmoid(av[m].astype(np.float32) * h["sc"])
            cls_sc = p[:, 5:].max(1)
            conf = p[:, 4] * cls_sc
            k = conf > args.conf
            if not k.any():
                continue
            p = p[k]; gy = gy[k]; gx = gx[k]; ai = ai[k]
            cx = (p[:,0]*2.0 - 0.5 + gx) * h["stride"]
            cy = (p[:,1]*2.0 - 0.5 + gy) * h["stride"]
            bw = (p[:,2]*2.0)**2 * h["anc"][ai, 0]
            bh = (p[:,3]*2.0)**2 * h["anc"][ai, 1]
            boxes.append(np.stack([cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2], 1))
            scores.append(conf[k])
            classes.append(p[:, 5:].argmax(1))

        if boxes:
            bx = np.concatenate(boxes, 0)
            sc_ = np.concatenate(scores, 0)
            cl = np.concatenate(classes, 0)
            bx[:,[0,2]] = ((bx[:,[0,2]] - dw) / r).clip(0, VW)
            bx[:,[1,3]] = ((bx[:,[1,3]] - dh) / r).clip(0, VH)
            fin = []
            for cc in np.unique(cl):
                idx = np.where(cl == cc)[0]
                fin.extend(idx[nms(bx[idx], sc_[idx], args.iou)])
            fin = np.array(sorted(fin, key=lambda i: -sc_[i]), dtype=int)
            bx, sc_, cl = bx[fin], sc_[fin], cl[fin]
        else:
            bx = np.zeros((0,4)); sc_ = np.zeros(0); cl = np.zeros(0, dtype=int)
        e = time.time()

        if writer is not None:
            for i in range(len(bx)):
                x1,y1,x2,y2 = [int(v) for v in bx[i]]
                nm = COCO[int(cl[i])] if int(cl[i]) < len(COCO) else "cls"
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,0), 3)
                cv2.putText(frame, "%s %.2f" % (nm, sc_[i]), (x1, max(0,y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,200,0), 3)
            writer.write(frame)

        if not warm:
            t_dec += b-a; t_pre += c-b; t_dpu += d-c; t_post += e-d
            ndet += len(bx)
        n += 1

    cap.release()
    if writer is not None:
        writer.release()

    m = n - args.warmup
    if m <= 0:
        print("no frames processed"); return 1
    wall = time.time() - t_wall0
    s = t_dec + t_pre + t_dpu + t_post
    tot = s / m * 1e3

    print("\nframes timed        : %d  (%d warm-up discarded)" % (m, args.warmup))
    print("detections total    : %d  (%.1f per frame)" % (ndet, ndet/m))
    print("\n--- mean ms per frame ---")
    print("  decode (MJPEG,CPU): %8.2f   %5.1f%%" % (t_dec/m*1e3, 100*t_dec/s))
    print("  preprocess  (LUT) : %8.2f   %5.1f%%" % (t_pre/m*1e3, 100*t_pre/s))
    print("  DPU inference     : %8.2f   %5.1f%%" % (t_dpu/m*1e3, 100*t_dpu/s))
    print("  postprocess (gated): %7.2f   %5.1f%%" % (t_post/m*1e3, 100*t_post/s))
    print("  ---------------------------------")
    print("  TOTAL             : %8.2f" % tot)
    print("\nEND-TO-END          : %.2f FPS   (pipeline sum)" % (1000.0/tot))
    print("wall-clock          : %.2f FPS   (%.1fs for %d frames)" % (m/wall, wall, m))
    print("\nDPU-only ceiling    : %.2f FPS  (if decode+pre+post were free)"
          % (m/t_dpu))
    return 0


if __name__ == "__main__":
    sys.exit(main())
