#!/usr/bin/env python3
"""
End-to-end VIDEO benchmark for the Vitis-AI 2.5 ofa_yolo models on the
stock kv260-benchmark-b4096 DPU.

Times every stage separately so the real bottleneck is visible:
    decode  -> preprocess (letterbox+quantise) -> DPU -> postprocess (decode+NMS)

  python3 bench_video.py <xmodel> <video> [--max-frames N] [--conf C] [--save out.avi]
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

ANCHORS = {8:  [(10,13),(16,30),(33,23)],
           16: [(30,61),(62,45),(59,119)],
           32: [(116,90),(156,198),(373,326)]}


def dpu_subgraph(graph):
    root = graph.get_root_subgraph()
    return [s for s in root.toposort_child_subgraph()
            if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]


def letterbox(img, new, color=114):
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


def postprocess(outdata, out_meta, size, conf_thr, iou_thr, ratio, dw, dh, W0, H0):
    boxes, scores, classes = [], [], []
    for arr, (dims, fp, sc) in zip(outdata, out_meta):
        Gy, Gx, C = dims[1], dims[2], dims[3]
        stride = size // Gy
        if stride not in ANCHORS:
            continue
        na = 3; no = C // na
        p = sigmoid(arr.astype(np.float32) * sc).reshape(Gy, Gx, na, no)
        gy, gx = np.meshgrid(np.arange(Gy), np.arange(Gx), indexing="ij")
        gx = gx[..., None]; gy = gy[..., None]
        cx = (p[...,0]*2.0 - 0.5 + gx) * stride
        cy = (p[...,1]*2.0 - 0.5 + gy) * stride
        aw = np.array([a[0] for a in ANCHORS[stride]], dtype=np.float32)
        ah = np.array([a[1] for a in ANCHORS[stride]], dtype=np.float32)
        bw = (p[...,2]*2.0)**2 * aw
        bh = (p[...,3]*2.0)**2 * ah
        cls = p[...,5:]
        conf = p[...,4] * cls.max(-1)
        m = conf > conf_thr
        if not m.any():
            continue
        bx, by, bwv, bhv = cx[m], cy[m], bw[m], bh[m]
        boxes.append(np.stack([bx-bwv/2, by-bhv/2, bx+bwv/2, by+bhv/2], 1))
        scores.append(conf[m]); classes.append(cls.argmax(-1)[m])
    if not boxes:
        return np.zeros((0,4)), np.zeros(0), np.zeros(0, dtype=int)
    boxes = np.concatenate(boxes,0); scores = np.concatenate(scores,0)
    classes = np.concatenate(classes,0)
    boxes[:,[0,2]] = ((boxes[:,[0,2]] - dw) / ratio).clip(0, W0)
    boxes[:,[1,3]] = ((boxes[:,[1,3]] - dh) / ratio).clip(0, H0)
    final = []
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        final.extend(idx[nms(boxes[idx], scores[idx], iou_thr)])
    final = np.array(sorted(final, key=lambda i: -scores[i]), dtype=int)
    return boxes[final], scores[final], classes[final]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xmodel"); ap.add_argument("video")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--save", default=None)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    g = xir.Graph.deserialize(args.xmodel)
    subs = dpu_subgraph(g)
    runner = vart.Runner.create_runner(subs[0], "run")
    it = runner.get_input_tensors(); ot = runner.get_output_tensors()
    in_dims = tuple(it[0].dims)
    in_scale = float(2 ** it[0].get_attr("fix_point"))
    size = in_dims[1]
    out_meta = [(tuple(t.dims), t.get_attr("fix_point"),
                 2.0 ** -t.get_attr("fix_point")) for t in ot]
    outdata = [np.empty(d, dtype=np.int8, order="C") for d,_,_ in out_meta]

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("cannot open video: %s" % args.video); return 1
    VW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    VH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("model : %s" % args.xmodel.split("/")[-1])
    print("video : %s  %dx%d" % (args.video.split("/")[-1], VW, VH))
    print("net   : %dx%d" % (size, size))

    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"MJPG"),
                                 30, (VW, VH))

    t_dec = t_pre = t_dpu = t_post = t_draw = 0.0
    n = 0; ndet = 0
    t_wall0 = None
    while n < args.max_frames + args.warmup:
        a = time.time()
        ok, frame = cap.read()
        b = time.time()
        if not ok:
            break
        warm = n < args.warmup
        if not warm and t_wall0 is None:
            t_wall0 = a

        lb, ratio, (dw, dh) = letterbox(frame, size)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        q = np.clip(np.round(rgb.astype(np.float32) / 255.0 * in_scale),
                    -128, 127).astype(np.int8)
        indata = [np.ascontiguousarray(q.reshape(in_dims))]
        c = time.time()

        jid = runner.execute_async(indata, outdata)
        runner.wait(jid)
        d = time.time()

        bx, sc, cl = postprocess(outdata, out_meta, size, args.conf, args.iou,
                                 ratio, dw, dh, VW, VH)
        e = time.time()

        if writer is not None:
            for i in range(len(bx)):
                x1,y1,x2,y2 = [int(v) for v in bx[i]]
                nm = COCO[int(cl[i])] if int(cl[i]) < len(COCO) else "cls"
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,0), 3)
                cv2.putText(frame, "%s %.2f" % (nm, sc[i]), (x1, max(0,y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,200,0), 3)
            writer.write(frame)
        f = time.time()

        if not warm:
            t_dec += b-a; t_pre += c-b; t_dpu += d-c; t_post += e-d; t_draw += f-e
            ndet += len(bx)
        n += 1

    cap.release()
    if writer is not None:
        writer.release()

    m = n - args.warmup
    if m <= 0:
        print("no frames processed"); return 1
    wall = time.time() - t_wall0
    tot = (t_dec + t_pre + t_dpu + t_post) / m * 1e3

    print("\nframes timed        : %d  (%d warm-up discarded)" % (m, args.warmup))
    print("detections total    : %d  (%.1f per frame)" % (ndet, ndet / m))
    print("\n--- mean ms per frame ---")
    print("  decode (MJPEG,CPU): %8.2f   %5.1f%%" % (t_dec/m*1e3, 100*t_dec/(t_dec+t_pre+t_dpu+t_post)))
    print("  preprocess        : %8.2f   %5.1f%%" % (t_pre/m*1e3, 100*t_pre/(t_dec+t_pre+t_dpu+t_post)))
    print("  DPU inference     : %8.2f   %5.1f%%" % (t_dpu/m*1e3, 100*t_dpu/(t_dec+t_pre+t_dpu+t_post)))
    print("  postprocess+NMS   : %8.2f   %5.1f%%" % (t_post/m*1e3, 100*t_post/(t_dec+t_pre+t_dpu+t_post)))
    print("  ---------------------------------")
    print("  TOTAL             : %8.2f" % tot)
    print("\nEND-TO-END          : %.2f FPS   (pipeline sum)" % (1000.0/tot))
    print("wall-clock          : %.2f FPS   (%.1fs for %d frames)" % (m/wall, wall, m))
    if args.save:
        print("annotated video     : %s  (draw+encode %.2f ms/frame, NOT in totals)"
              % (args.save, t_draw/m*1e3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
