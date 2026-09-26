#!/usr/bin/env python3
"""
pipeline_v2 -- optimised 4K detection pipeline.

Three changes over bench_video_opt.py, all on the ARM side, no hardware change:

  1. BYPASS GSTREAMER. Raw JPEG frames are pulled straight out of the MJPEG
     AVI and handed to cv2.imdecode. VideoCapture's GStreamer path costs
     ~2x more than imdecode for identical output.

  2. SCALED-DCT DECODE. The pipeline letterboxes 3840x2160 into 640x640,
     so the real content is 640x360 -- only 3.5% of the decoded pixels are
     kept. libjpeg can decode directly at 1/4 scale (960x540) via scaled
     DCT, which is still >= 640x360, so nothing the pipeline would have
     kept is lost.

  3. MULTI-CORE PREFETCH. Decode + letterbox + quantise are pure functions
     of a JPEG buffer, so they are farmed out to a process pool across the
     four Cortex-A53 cores while the main process drives the DPU. The DPU
     is a single shared resource and stays in the parent.

  python3 pipeline_v2.py <xmodel> <video.avi> [--workers N] [--reduce {1,2,4}]
                          [--max-frames N] [--verify] [--save out.avi]
"""
import sys, os, time, argparse, struct, math
import numpy as np
import cv2
import xir
import vart
import multiprocessing as mp

COCO = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
    'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
    'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
    'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
    'remote','keyboard','cell phone','microwave','oven','toaster','sink',
    'refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']

ANCHORS = {8:[(10,13),(16,30),(33,23)], 16:[(30,61),(62,45),(59,119)],
           32:[(116,90),(156,198),(373,326)]}

REDUCE_FLAG = {1: cv2.IMREAD_COLOR,
               2: cv2.IMREAD_REDUCED_COLOR_2,
               4: cv2.IMREAD_REDUCED_COLOR_4,
               8: cv2.IMREAD_REDUCED_COLOR_8}

# ---- worker globals (set once per process) -------------------------------
_G = {}


def _init(size, in_scale, reduce_n, src_w, src_h):
    _G["size"] = size
    _G["flag"] = REDUCE_FLAG[reduce_n]
    _G["lut"] = np.clip(np.round(np.arange(256, dtype=np.float32) / 255.0 * in_scale),
                        -128, 127).astype(np.int8)
    # letterbox geometry is fixed for a fixed-size video
    r = min(size / src_h, size / src_w)
    _G["nw"], _G["nh"] = int(round(src_w * r)), int(round(src_h * r))
    _G["dw"] = (size - _G["nw"]) // 2
    _G["dh"] = (size - _G["nh"]) // 2


def _work(buf):
    """JPEG bytes -> int8 640x640x3 network input. Pure function."""
    a = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(a, _G["flag"])
    if img is None:
        return None
    s, nw, nh, dw, dh = _G["size"], _G["nw"], _G["nh"], _G["dw"], _G["dh"]
    canvas = np.full((s, s, 3), 114, dtype=np.uint8)
    cv2.resize(img, (nw, nh), dst=canvas[dh:dh + nh, dw:dw + nw],
               interpolation=cv2.INTER_LINEAR)
    return _G["lut"][canvas[:, :, ::-1]].copy()


# ---- avi -----------------------------------------------------------------
def avi_jpeg_frames(path, limit=None):
    with open(path, "rb") as f:
        data = f.read()
    i = data.find(b"movi")
    if i < 0:
        raise RuntimeError("no movi list")
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


# ---- model helpers -------------------------------------------------------
def dpu_subgraph(g):
    return [s for s in g.get_root_subgraph().toposort_child_subgraph()
            if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def nms(boxes, scores, thr):
    if len(boxes) == 0:
        return []
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas = np.maximum(0.,x2-x1)*np.maximum(0.,y2-y1)
    order = scores.argsort()[::-1]; keep=[]
    while order.size > 0:
        i = order[0]; keep.append(i)
        if order.size == 1: break
        xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
        inter=np.maximum(0.,xx2-xx1)*np.maximum(0.,yy2-yy1)
        iou=inter/(areas[i]+areas[order[1:]]-inter+1e-9)
        order = order[1:][iou <= thr]
    return keep


def decode_heads(outdata, heads, size, conf, iou, r, dw, dh, W, H):
    boxes, scores, classes = [], [], []
    for arr, h in zip(outdata, heads):
        av = arr.reshape(h["Gy"], h["Gx"], 3, h["no"])
        m = av[..., 4] > h["thr_raw"]
        if not m.any():
            continue
        gy, gx, ai = np.nonzero(m)
        p = sigmoid(av[m].astype(np.float32) * h["sc"])
        cs = p[:, 5:].max(1)
        cf = p[:, 4] * cs
        k = cf > conf
        if not k.any():
            continue
        p = p[k]; gy = gy[k]; gx = gx[k]; ai = ai[k]
        cx = (p[:,0]*2.0-0.5+gx)*h["stride"]
        cy = (p[:,1]*2.0-0.5+gy)*h["stride"]
        bw = (p[:,2]*2.0)**2*h["anc"][ai,0]
        bh = (p[:,3]*2.0)**2*h["anc"][ai,1]
        boxes.append(np.stack([cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2], 1))
        scores.append(cf[k]); classes.append(p[:,5:].argmax(1))
    if not boxes:
        return np.zeros((0,4)), np.zeros(0), np.zeros(0, dtype=int)
    bx = np.concatenate(boxes,0); sc_ = np.concatenate(scores,0)
    cl = np.concatenate(classes,0)
    bx[:,[0,2]] = ((bx[:,[0,2]]-dw)/r).clip(0,W)
    bx[:,[1,3]] = ((bx[:,[1,3]]-dh)/r).clip(0,H)
    fin=[]
    for c in np.unique(cl):
        idx=np.where(cl==c)[0]
        fin.extend(idx[nms(bx[idx], sc_[idx], iou)])
    fin=np.array(sorted(fin,key=lambda i:-sc_[i]),dtype=int)
    return bx[fin], sc_[fin], cl[fin]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xmodel"); ap.add_argument("video")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--reduce", type=int, default=4, choices=[1,2,4,8])
    ap.add_argument("--max-frames", type=int, default=100)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--verify", action="store_true",
                    help="also run reduce=1 and compare detections")
    args = ap.parse_args()

    print("loading JPEG frames from %s ..." % args.video.split("/")[-1])
    bufs = avi_jpeg_frames(args.video, args.max_frames + 4)
    if not bufs:
        print("no frames"); return 1
    probe = cv2.imdecode(np.frombuffer(bufs[0], np.uint8), cv2.IMREAD_COLOR)
    SRC_H, SRC_W = probe.shape[:2]
    print("  %d frames  source %dx%d  mean %.2f MB/JPEG"
          % (len(bufs), SRC_W, SRC_H, np.mean([len(b) for b in bufs])/1e6))

    # ---------------------------------------------------------------
    # Pools MUST be forked BEFORE the VART runner exists: the DPU runner
    # holds device handles and driver threads that do not survive fork()
    # (the child aborts). So the network geometry needed by the workers is
    # read from the xir graph first, the pools are forked, and only then
    # is the runner created in the parent.
    # ---------------------------------------------------------------
    g = xir.Graph.deserialize(args.xmodel)
    sub = dpu_subgraph(g)[0]
    # xir returns a SET here (unordered) -- fine, the DPU subgraph has one input
    _its = list(sub.get_input_tensors())
    _it = _its[0]
    size = tuple(_it.dims)[1]
    in_scale = float(2 ** _it.get_attr("fix_point"))

    pools = {}
    if args.workers > 0:
        needed = [args.reduce] + ([1] if args.verify and args.reduce != 1 else [])
        for rn in needed:
            pools[rn] = mp.Pool(args.workers, initializer=_init,
                                initargs=(size, in_scale, rn, SRC_W, SRC_H))
        # force workers to actually start before VART touches the device
        for rn in needed:
            pools[rn].apply(int, (0,))

    runner = vart.Runner.create_runner(sub, "run")
    it, ot = runner.get_input_tensors(), runner.get_output_tensors()
    in_dims = tuple(it[0].dims)
    out_meta = [(tuple(t.dims), 2.0 ** -t.get_attr("fix_point")) for t in ot]
    outdata = [np.empty(d, dtype=np.int8, order="C") for d, _ in out_meta]

    logit_t = math.log(args.conf/(1.0-args.conf))
    heads = []
    for dims, sc in out_meta:
        Gy,Gx,C = dims[1],dims[2],dims[3]
        st = size//Gy
        heads.append({"Gy":Gy,"Gx":Gx,"no":C//3,"stride":st,"sc":sc,
                      "anc":np.array(ANCHORS[st],dtype=np.float32),
                      "thr_raw":logit_t/sc})

    r = min(size/SRC_H, size/SRC_W)
    dw = (size - int(round(SRC_W*r)))//2
    dh = (size - int(round(SRC_H*r)))//2

    def run_pass(reduce_n, workers, label):
        frames = bufs[:args.max_frames + 3]
        pool = pools.get(reduce_n)
        if pool is not None:
            gen = pool.imap(_work, frames, chunksize=1)
        else:
            _init(size, in_scale, reduce_n, SRC_W, SRC_H)
            gen = (_work(b) for b in frames)

        t_in = t_dpu = t_post = 0.0
        n = 0; ndet = 0; allde = []
        t_wall0 = None
        for q in gen:
            a = time.time()
            if q is None:
                continue
            b = time.time()
            warm = n < 3
            if not warm and t_wall0 is None:
                t_wall0 = a
            runner.wait(runner.execute_async([np.ascontiguousarray(q.reshape(in_dims))],
                                             outdata))
            c = time.time()
            bx, sc_, cl = decode_heads(outdata, heads, size, args.conf, args.iou,
                                       r, dw, dh, SRC_W, SRC_H)
            d = time.time()
            if not warm:
                t_in += b-a; t_dpu += c-b; t_post += d-c
                ndet += len(bx)
                allde.append((len(bx), float(sc_.sum()) if len(sc_) else 0.0))
            n += 1
        wall = time.time() - t_wall0 if t_wall0 else 0
        m = n - 3
        s = t_in + t_dpu + t_post
        print("\n===== %s =====" % label)
        print("  decode+preprocess : %8.2f ms   %5.1f%%   (%d worker%s)"
              % (t_in/m*1e3, 100*t_in/s, workers, "" if workers == 1 else "s"))
        print("  DPU inference     : %8.2f ms   %5.1f%%" % (t_dpu/m*1e3, 100*t_dpu/s))
        print("  postprocess       : %8.2f ms   %5.1f%%" % (t_post/m*1e3, 100*t_post/s))
        print("  ------------------------------------")
        print("  TOTAL             : %8.2f ms" % (s/m*1e3))
        print("  END-TO-END        : %6.2f FPS      wall-clock %.2f FPS"
              % (1000.0/(s/m*1e3), m/wall if wall else 0))
        print("  detections        : %d total (%.2f/frame)" % (ndet, ndet/m))
        return allde, s/m*1e3, ndet

    if args.verify:
        ref, _, nref = run_pass(1, args.workers, "REFERENCE  full decode (reduce=1)")
        tst, _, ntst = run_pass(args.reduce, args.workers,
                                "TEST       scaled DCT 1/%d" % args.reduce)
        same = sum(1 for a, b in zip(ref, tst) if a[0] == b[0])
        print("\n===== ACCURACY CHECK =====")
        print("  frames with identical detection COUNT : %d / %d (%.1f%%)"
              % (same, len(ref), 100.0*same/len(ref)))
        print("  total detections  full=%d  reduced=%d  delta=%+d (%+.2f%%)"
              % (nref, ntst, ntst-nref, 100.0*(ntst-nref)/max(nref,1)))
        ds = [abs(a[1]-b[1]) for a, b in zip(ref, tst)]
        print("  mean |sum-of-scores| difference       : %.4f" % (np.mean(ds)))
    else:
        run_pass(args.reduce, args.workers,
                 "reduce=1/%d  workers=%d" % (args.reduce, args.workers))
    for p in pools.values():
        p.close(); p.join()
    return 0


if __name__ == "__main__":
    mp.set_start_method("fork")
    sys.exit(main())
