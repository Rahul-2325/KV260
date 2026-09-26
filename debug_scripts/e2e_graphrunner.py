#!/usr/bin/env python3
"""
e2e_graphrunner.py  -- PHASE 1: end-to-end inference baseline

Runs the FULL lcam_v5.xmodel using vitis_ai_library.GraphRunner, which
dispatches DPU subgraphs to the DPU and CPU subgraphs (the LCAM attention)
to the ARM core automatically.

Produces the numbers needed for the paper's baseline row:
  - end-to-end latency / FPS (warm, averaged over N runs)
  - input/output tensor geometry
  - raw output tensors saved for the postprocessing step

Usage:
  export XLNX_VART_FIRMWARE=/lib/firmware/xilinx/kv260-benchmark-b4096/kv260-benchmark-b4096.xclbin
  python3 e2e_graphrunner.py --image WEB09971.jpg --runs 20
"""
import argparse, os, sys, time
import numpy as np

XMODEL = '/home/root/lcam_v5.xmodel'
INPUT_W = INPUT_H = 640


def letterbox(img, w=INPUT_W, h=INPUT_H):
    import cv2
    oh, ow = img.shape[:2]
    s = min(w / ow, h / oh)
    nw, nh = int(ow * s), int(oh * s)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((h, w, 3), 114, dtype=np.uint8)
    px, py = (w - nw) // 2, (h - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    return canvas, s, px, py


def main():
    global XMODEL
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', default='/home/root/WEB09971.jpg')
    ap.add_argument('--xmodel', default=XMODEL,
                    help='compiled xmodel whose dpu_fingerprint MATCHES the '
                         'loaded DPU (see PROJECT_HISTORY.md §29)')
    ap.add_argument('--runs', type=int, default=20)
    ap.add_argument('--save', default='/home/root/e2e_outputs.npz')
    args = ap.parse_args()
    XMODEL = args.xmodel

    import cv2, xir
    from vitis_ai_library import GraphRunner

    print("=" * 66)
    print("PHASE 1: END-TO-END BASELINE (DPU + CPU fallback)")
    print("=" * 66)
    print("xclbin (XLNX_VART_FIRMWARE) = %s" % os.environ.get('XLNX_VART_FIRMWARE', '<unset>'))

    g = xir.Graph.deserialize(XMODEL)
    t0 = time.perf_counter()
    runner = GraphRunner.create_graph_runner(g)
    print("GraphRunner created in %.2f s" % (time.perf_counter() - t0))

    ins = runner.get_inputs()
    outs = runner.get_outputs()
    print("\ninput tensor buffers : %d" % len(ins))
    for i, tb in enumerate(ins):
        print("   [%d] %-44s %s" % (i, tb.get_tensor().name[:44], list(tb.get_tensor().dims)))
    print("output tensor buffers: %d" % len(outs))
    for i, tb in enumerate(outs):
        print("   [%d] %-44s %s" % (i, tb.get_tensor().name[:44], list(tb.get_tensor().dims)))

    # ---- preprocess -------------------------------------------------------
    img = cv2.imread(args.image)
    if img is None:
        print("ERROR: cannot read %s" % args.image); sys.exit(1)
    oh, ow = img.shape[:2]
    canvas, scale, px, py = letterbox(img)
    print("\nimage %s  %dx%d -> letterboxed 640x640 (scale=%.4f pad=%d,%d)"
          % (os.path.basename(args.image), ow, oh, scale, px, py))

    in_arr = np.asarray(ins[0])          # shape (1,H,W,C), dtype per tensor
    print("input buffer dtype=%s shape=%s" % (in_arr.dtype, in_arr.shape))

    # quantised input: scale by 2^fix_point if the tensor is fixed-point
    fp = None
    t = ins[0].get_tensor()
    for k in ('fix_point', 'fix_pos'):
        if t.has_attr(k):
            fp = t.get_attr(k); break
    print("input fix_point = %s" % fp)

    # QUANTISATION (this is easy to get wrong -- see PROJECT_HISTORY §29.8):
    #   real_value = int8_value * 2^(-fix_point)
    #   => int8_value = real_value * 2^(fix_point)
    # YOLOX consumes RAW 0..255 BGR pixels (it does NOT normalise to [0,1]),
    # so real_value == pixel. With fix_point = -1 that gives pixel/2 -> 0..127,
    # which fits int8 exactly. Dividing by 255 first collapses everything to
    # zero and produces ~uniform garbage scores.
    data = canvas.astype(np.float32)          # raw pixels, NOT /255
    if fp is not None and np.issubdtype(in_arr.dtype, np.integer):
        data = np.clip(np.round(data * (2.0 ** fp)), -128, 127)
    in_arr[0] = data.astype(in_arr.dtype)
    print("quantised input range: [%d, %d]" % (in_arr.min(), in_arr.max()))

    # ---- run --------------------------------------------------------------
    print("\nwarming up...")
    for _ in range(3):
        jid = runner.execute_async(ins, outs)
        runner.wait(jid)

    print("timing %d runs..." % args.runs)
    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        jid = runner.execute_async(ins, outs)
        runner.wait(jid)
        times.append(time.perf_counter() - t0)
    times = np.array(times) * 1000.0

    print("\n" + "=" * 66)
    print("RESULT -- full model, DPU + CPU fallback")
    print("=" * 66)
    print("  mean latency : %8.2f ms" % times.mean())
    print("  median       : %8.2f ms" % np.median(times))
    print("  min / max    : %8.2f / %.2f ms" % (times.min(), times.max()))
    print("  std          : %8.2f ms" % times.std())
    print("  FPS (mean)   : %8.2f" % (1000.0 / times.mean()))
    print("=" * 66)

    # ---- save raw outputs for postprocessing ------------------------------
    d = {}
    for i, tb in enumerate(outs):
        a = np.array(np.asarray(tb))
        d['out_%d' % i] = a
        d['out_%d_name' % i] = np.array(tb.get_tensor().name)
    d['scale'] = np.array(scale); d['pad'] = np.array([px, py])
    d['orig'] = np.array([ow, oh]); d['latency_ms'] = times
    np.savez(args.save, **d)
    print("\nraw outputs + meta saved to %s" % args.save)
    for i, tb in enumerate(outs):
        a = np.asarray(tb)
        print("   out_%d shape=%-20s dtype=%s" % (i, list(a.shape), a.dtype))


if __name__ == '__main__':
    main()
