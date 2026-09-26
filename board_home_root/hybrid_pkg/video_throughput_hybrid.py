#!/usr/bin/env python3
"""
video_throughput_hybrid.py -- run ON THE BOARD, kv260-hybrid2/3 loaded.

Measures TRUE camera-to-screen throughput of the LCAM-YOLOX HYBRID pipeline
(DPU + the custom LCAM accelerator) on genuine fire/smoke video -- not a
single repeated still image (pipelined_throughput.py measures compute only)
and NOT the stock DPU+GraphRunner path.

Two other scripts already on this board -- lcam_video_4k.py and
tiled_lcam.py -- default to XLNX_VART_FIRMWARE=kv260-benchmark-b4096, the
STOCK bitstream with no custom accelerator, and drive the model through
vitis_ai_library.GraphRunner (the slow CPU-fallback path, ~427 ms/frame,
see PROJECT_HISTORY section 33). They are a different track's artifacts and
are NOT reused here. This script uses the exact same hand-written engine as
hybrid_pipeline.py / pipelined_throughput.py: DPU subgraphs on the DPU, the
four attention gates on the LCAM accelerator at 0xa0020000, everything else
in numpy.

Every stage is timed and printed SEPARATELY, never folded into one number
without saying so (the convention this whole project has followed since
section 33.4):
  decode      -- cv2.VideoCapture read
  preprocess  -- letterbox + int8 quantise
  DPU         -- 8 subgraphs (vendor IP)
  LCAM gate   -- 4 attention gates on the custom accelerator
  other CPU   -- remaining glue subgraphs (LUT/int8-fast-pathed, section 34/36)
  postprocess -- decode + NMS, with the concat order fixed ONCE before the
                 loop (searching it per frame, as the one-shot --out demo
                 does, would waste ~15 ms/frame for no reason on a fixed model)
  draw+encode -- ONLY charged if --save is given; a real annotated video
                 costs real MJPEG encode time, and the throughput number
                 without --save is the honest inference+decode ceiling

REQUIRED environment (identical to hybrid_pipeline.py):
  XLNX_VART_FIRMWARE=/lib/firmware/xilinx/kv260-hybrid2/hybrid.xclbin
  XRT_INI_PATH=/home/root/xrt.ini

This script REFUSES to run if the loaded DPU fingerprint is not
0x101000012010407 (kv260-hybrid2/3) -- specifically because it is easy to
reach for lcam_video_4k.py by name-similarity and get the stock bitstream's
numbers by accident. A fingerprint mismatch inside VART is a hard C++ abort,
not a catchable Python exception, so the check happens up front via
`xdputil query` instead.
"""
import argparse
import subprocess
import sys
import time

import cv2
import numpy as np
import vart
import xir

sys.path.insert(0, "/home/root")
sys.path.insert(0, "/home/root/hybrid_pkg")
from hybrid_pipeline import (LcamCU, run_cpu_subgraph, build_lut,
                             build_int8_path, sg_out, is_gate, gate_params,
                             letterbox, XMODEL)
from postprocess_detections import build_grid, decode, sigmoid

EXPECTED_FP = "0x101000012010407"


def check_fingerprint():
    try:
        out = subprocess.check_output(["xdputil", "query"], text=True, timeout=10)
    except Exception as e:
        print("WARNING: could not run xdputil query (%s) -- proceeding anyway" % e)
        return
    if EXPECTED_FP not in out:
        print("FATAL: loaded DPU fingerprint is not %s" % EXPECTED_FP)
        print("       The hybrid bitstream (kv260-hybrid2/3) is NOT loaded --")
        print("       running would either hang or silently use the wrong design.")
        print("       Fix:  xmutil unloadapp; xmutil loadapp kv260-hybrid2")
        sys.exit(1)
    print("fingerprint check OK: %s (hybrid bitstream loaded)" % EXPECTED_FP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--gates", choices=("ip", "numpy"), default="ip")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole clip")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--nms", type=float, default=0.45)
    ap.add_argument("--classes", default="fire,smoke")
    ap.add_argument("--save", default=None,
                    help="write an annotated MJPG video here (adds real "
                         "draw+encode cost, reported separately)")
    args = ap.parse_args()

    check_fingerprint()

    g = xir.Graph.deserialize(XMODEL)
    subs = g.get_root_subgraph().toposort_child_subgraph()
    kinds, luts = {}, {}
    for i, s in enumerate(subs):
        kinds[i] = s.get_attr("device").upper() if s.has_attr("device") else "USER"
        if kinds[i] == "CPU" and not is_gate(s):
            r = build_lut(s)
            if r is not None:
                luts[i] = ("lut",) + r
            else:
                r = build_int8_path(s)
                if r is not None:
                    luts[i] = ("int8",) + r

    runners = {i: vart.Runner.create_runner(s, "run")
              for i, s in enumerate(subs) if kinds[i] == "DPU"}
    cu = LcamCU() if args.gates == "ip" else None
    print("engine ready: %d DPU subgraphs, %d LUT/int8-fast CPU subgraphs, gates=%s"
          % (len(runners), len(luts), args.gates))

    in_name = sg_out(subs[0])[0].name
    in_fp = sg_out(subs[0])[0].get_attr("fix_point")
    out_name = sg_out(subs[-1])[0].name

    # PIXEL QUANTISATION AS A 256-ENTRY LUT.
    #
    # First run of this script measured "preprocess" (letterbox + quantise)
    # at 32.12 ms/frame -- 27% of the total, and a genuinely NEW cost: every
    # earlier benchmark in this project (hybrid_pipeline.py,
    # pipelined_throughput.py) quantised ONE image once before its timing
    # loop, so this per-frame cost was never measured before now.
    #
    # in_fp is a fixed property of the compiled xmodel (it does not vary
    # frame to frame), so `clip(round(pixel * 2**in_fp))` is exactly the
    # same 256-entry mapping every time -- computed here ONCE and applied
    # by indexing, instead of running astype(float32)+multiply+round+clip+
    # astype(int8) over a 640x640x3 = 1.2M-element array on every frame.
    # This is the identical technique used in 4K_VIDEO_RESEARCH.md section
    # 13.1 for the same reason, on the same class of ARM-core cost.
    PIX_LUT = np.clip(np.round(np.arange(256, dtype=np.float32) * (2.0 ** in_fp)),
                      -128, 127).astype(np.int8)
    # Verify bit-exactness against the original formula ONCE, on real data,
    # rather than trusting the algebra -- the whole project's convention.
    _test = np.arange(256, dtype=np.uint8)
    _ref = np.clip(np.round(_test.astype(np.float32) * (2.0 ** in_fp)),
                   -128, 127).astype(np.int8)
    assert np.array_equal(PIX_LUT[_test], _ref), "quantisation LUT is not bit-exact"

    # Fixed, empirically-verified concat order (postprocess_detections.py).
    # Computed ONCE outside the loop -- this is a property of the compiled
    # model, not of the frame, so re-deriving it 120 times would be pure
    # waste (it is exactly the ~15 ms/frame "decode" cost measured in
    # section 41 for the one-shot demo path, which searches both orders
    # every single call because it only ever runs once).
    grids, strides = build_grid([(80, 8), (40, 16), (20, 32)])
    names = args.classes.split(",")

    cap = cv2.VideoCapture(args.video)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        nframes = min(nframes, args.max_frames)
    print("video: %s   %d frames to process" % (args.video, nframes))

    writer = None  # opened lazily once the real frame size is known

    stages = {"decode": 0.0, "pre": 0.0, "dpu": 0.0, "gate": 0.0,
             "cpu": 0.0, "post": 0.0, "draw": 0.0}
    total_dets = 0
    per_frame_dets = []
    n = 0
    t_wall0 = time.perf_counter()

    for _ in range(nframes):
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        stages["decode"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        canvas, scale, px, py, ow, oh = letterbox(frame)
        quant = PIX_LUT[canvas].reshape(1, 640, 640, 3)
        stages["pre"] += time.perf_counter() - t0

        vals = {in_name: quant}
        for i, s in enumerate(subs):
            k = kinds[i]
            if k == "USER":
                continue
            if k == "DPU":
                r = runners[i]
                its, ots = r.get_input_tensors(), r.get_output_tensors()
                ins = [np.ascontiguousarray(vals[t.name]) for t in its]
                outs = [np.zeros(tuple(t.dims), dtype=np.int8) for t in ots]
                t0 = time.perf_counter()
                r.wait(r.execute_async(ins, outs))
                stages["dpu"] += time.perf_counter() - t0
                for t, o in zip(ots, outs):
                    vals[t.name] = o
            elif cu is not None and is_gate(s):
                ft, wt, fpf, fpw, fpo = gate_params(s)
                t0 = time.perf_counter()
                vals[sg_out(s)[0].name] = cu.gate(vals[ft], vals[wt], fpf, fpw, fpo)
                stages["gate"] += time.perf_counter() - t0
            elif i in luts:
                kind, src, dst, f = luts[i]
                t0 = time.perf_counter()
                vals[dst] = (f[vals[src].view(np.uint8)] if kind == "lut"
                            else f(vals[src]))
                stages["cpu"] += time.perf_counter() - t0
            else:
                t0 = time.perf_counter()
                run_cpu_subgraph(s, vals)
                stages["cpu"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        pred = np.asarray(vals[out_name], dtype=np.float32)
        while pred.ndim > 2:
            pred = pred[0]
        if pred.shape[0] == 7:
            pred = pred.T
        cxcy, wh = decode(pred, grids, strides)
        raw_obj, raw_cls = pred[:, 4], pred[:, 5:]
        already = (raw_obj.min() >= 0 and raw_obj.max() <= 1
                  and raw_cls.min() >= 0 and raw_cls.max() <= 1)
        obj, cls = (raw_obj, raw_cls) if already else (sigmoid(raw_obj), sigmoid(raw_cls))
        scores = obj[:, None] * cls
        cid, csc = scores.argmax(1), scores.max(1)
        keep = csc > args.conf
        dets = []
        if keep.sum():
            b = np.concatenate([cxcy[keep] - wh[keep] / 2, cxcy[keep] + wh[keep] / 2], 1)
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
                for q in (np.array(idx).flatten() if len(idx) else []):
                    dets.append((names[k] if k < len(names) else "cls%d" % k,
                                float(ss[q]), tuple(bb[q].astype(int))))
        stages["post"] += time.perf_counter() - t0
        total_dets += len(dets)
        per_frame_dets.append(len(dets))

        if args.save:
            t0 = time.perf_counter()
            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                src_fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
                writer = cv2.VideoWriter(args.save, fourcc, src_fps, (w, h))
            for nm, sc, (x1, y1, x2, y2) in dets:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "%s %.2f" % (nm, sc), (x1, max(y1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            writer.write(frame)
            stages["draw"] += time.perf_counter() - t0

        n += 1
        if n % 50 == 0:
            print("  %d / %d frames  (%.1f ms/frame so far, %d detections so far)"
                  % (n, nframes, (time.perf_counter() - t_wall0) * 1e3 / n, total_dets))

    wall = time.perf_counter() - t_wall0
    cap.release()
    if writer:
        writer.release()

    print()
    print("=" * 64)
    print("frames processed              : %d   in %.2f s" % (n, wall))
    print("-" * 64)
    for k, label in (("decode", "decode"), ("pre", "preprocess"),
                     ("dpu", "DPU (8 subgraphs)"), ("gate", "LCAM gate (4)"),
                     ("cpu", "other CPU subgraphs"), ("post", "postprocess (decode+NMS)"),
                     ("draw", "draw+encode (--save only)")):
        if k == "draw" and not args.save:
            continue
        print("  %-26s : %7.2f ms/frame  (%4.1f%% of wall clock)"
              % (label, stages[k] * 1e3 / n, 100 * stages[k] / wall))
    print("-" * 64)
    print("END-TO-END, camera to boxes   : %7.2f ms/frame -> %5.2f FPS"
          % (wall * 1e3 / n, n / wall))
    print("total detections in this clip : %d   (%.2f / frame, %d/%d frames with >=1)"
          % (total_dets, total_dets / n, sum(1 for d in per_frame_dets if d > 0), n))
    print("=" * 64)
    print("for comparison (compute-only, no video decode/postprocess):")
    print("  single-frame hardware pipeline   : 75.8 ms  -> 13.2 FPS  (sections 33-40)")
    print("  pipelined hardware, 4 workers     : 51.7 ms  -> 19.3 FPS  (sections 38, 41)")
    if args.save:
        print("annotated video written to: %s" % args.save)

    if cu:
        cu.close()


if __name__ == "__main__":
    main()
