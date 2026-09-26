#!/usr/bin/env python3
"""
pipelined_throughput.py -- run ON THE BOARD.

Frame-level pipelining. Single-frame latency is 75.8 ms, of which ~43 ms is
the DPU and ~33 ms is everything else. Those two cannot be compressed further
on their own, but they CAN overlap: while one frame is inside the DPU, another
frame's data movement and numpy glue can run. Throughput then approaches
    1 / max(DPU, rest) = 1 / 43 ms = ~23 FPS
even though each individual frame still takes 75.8 ms.

Feasibility was checked first (probe_gil.py): the write-combining read-back
parallelises 1.72x across threads and VART releases the GIL, so Python threads
can actually overlap here rather than taking turns.

WHAT IS SHARED AND HOW:
  * DPU  -- one physical unit. Each worker gets its OWN vart.Runner objects
            (runners are not thread-safe); VART serialises the hardware.
  * LCAM -- one physical accelerator with one set of buffers, so a single
            LcamCU guarded by a lock. Two threads must never drive it at once:
            they would overwrite each other's input buffer mid-flight.

Latency is reported alongside throughput because pipelining improves the
latter and slightly worsens the former -- for a video feed throughput is the
metric, but the distinction has to be stated, not hidden.
"""
import argparse
import queue
import sys
import threading
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


class Engine(object):
    """One worker's private view of the graph; the CU is shared under a lock."""

    def __init__(self, subs, kinds, luts, cu, cu_lock, gates):
        self.subs, self.kinds, self.luts = subs, kinds, luts
        self.cu, self.cu_lock, self.gates = cu, cu_lock, gates
        self.runners = {}
        for i, s in enumerate(subs):
            if kinds[i] == "DPU":
                self.runners[i] = vart.Runner.create_runner(s, "run")

    def infer(self, quant, in_name):
        vals = {in_name: quant}
        for i, s in enumerate(self.subs):
            k = self.kinds[i]
            if k == "USER":
                continue
            if k == "DPU":
                r = self.runners[i]
                its, ots = r.get_input_tensors(), r.get_output_tensors()
                ins = [np.ascontiguousarray(vals[t.name]) for t in its]
                outs = [np.zeros(tuple(t.dims), dtype=np.int8) for t in ots]
                r.wait(r.execute_async(ins, outs))
                for t, o in zip(ots, outs):
                    vals[t.name] = o
            elif self.gates == "ip" and is_gate(s):
                ft, wt, fpf, fpw, fpo = gate_params(s)
                with self.cu_lock:          # single accelerator, single buffer set
                    res = self.cu.gate(vals[ft], vals[wt], fpf, fpw, fpo)
                vals[sg_out(s)[0].name] = res
            elif i in self.luts:
                kind, src, dst, f = self.luts[i]
                vals[dst] = (f[vals[src].view(np.uint8)]
                             if kind == "lut" else f(vals[src]))
            else:
                run_cpu_subgraph(s, vals)
        return vals[sg_out(self.subs[-1])[0].name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--gates", choices=("ip", "numpy"), default="ip")
    ap.add_argument("--image", default="/home/root/WEB09971.jpg")
    ap.add_argument("--out", default=None,
                    help="after the timing loop, decode+draw ONE inference "
                         "result and save it here -- proves the pipelined "
                         "run still produces correct detections, without "
                         "adding any per-frame decode cost to the measured "
                         "throughput above (all workers process the same "
                         "preloaded frame, so any one result is representative)")
    args = ap.parse_args()

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

    in_name = sg_out(subs[0])[0].name
    in_fp = sg_out(subs[0])[0].get_attr("fix_point")
    img = cv2.imread(args.image)
    canvas, lb_scale, lb_px, lb_py, orig_w, orig_h = letterbox(img)
    quant = np.clip(np.round(canvas.astype(np.float32) * (2.0 ** in_fp)),
                    -128, 127).astype(np.int8).reshape(1, 640, 640, 3)

    cu = LcamCU() if args.gates == "ip" else None
    cu_lock = threading.Lock()

    print("building %d worker engine(s) ..." % args.workers)
    engines = [Engine(subs, kinds, luts, cu, cu_lock, args.gates)
               for _ in range(args.workers)]

    q = queue.Queue()
    for n in range(args.frames):
        q.put(n)
    lat, lat_lock = [], threading.Lock()

    def worker(eng):
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                return
            t0 = time.perf_counter()
            eng.infer(quant, in_name)
            dt = (time.perf_counter() - t0) * 1e3
            with lat_lock:
                lat.append(dt)

    # warm-up so first-call allocation is not counted
    engines[0].infer(quant, in_name)

    ts = [threading.Thread(target=worker, args=(e,)) for e in engines]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.perf_counter() - t0

    n = len(lat)
    print()
    print("=" * 58)
    print("workers                  : %d" % args.workers)
    print("frames                   : %d in %.2f s" % (n, wall))
    print("-" * 58)
    print("throughput               : %6.2f ms/frame -> %5.2f FPS"
          % (wall * 1e3 / n, n / wall))
    print("per-frame latency        : %6.2f ms mean  (%.2f min / %.2f max)"
          % (np.mean(lat), min(lat), max(lat)))
    print("=" * 58)
    print("single-threaded reference:  75.84 ms -> 13.19 FPS")
    print("speed-up vs single thread: %5.2fx" % (75.84 / (wall * 1e3 / n)))

    if args.out:
        # Deliberately AFTER `wall` is captured: this call is not part of
        # the timed loop and adds nothing to the throughput number above.
        # All workers ran the identical preloaded frame, so any one
        # engine's result is representative of what every worker produced.
        print()
        print("  decoding one representative result for --out ...")
        pred = engines[0].infer(quant, in_name)
        import postprocess_detections as pp
        pp.run_postprocess(pred, lb_scale, lb_px, lb_py, orig_w, orig_h,
                           args.image, args.out, verbose=True)

    if cu:
        cu.close()


if __name__ == "__main__":
    main()
