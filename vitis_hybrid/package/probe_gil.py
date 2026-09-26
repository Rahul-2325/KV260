#!/usr/bin/env python3
"""
probe_gil.py -- run ON THE BOARD.

Frame-level pipelining would overlap frame N's data movement with frame N+1's
DPU compute, raising throughput to 1/max(DPU, rest) = ~23 FPS. In Python that
only works if the expensive operations RELEASE THE GIL; if they hold it, two
threads simply take turns and nothing is gained.

Two costs dominate a frame and are tested separately here:
  * VART execute_async + wait     43.0 ms/frame   (a C++ call; releases only
                                                   if the binding says so)
  * the write-combining read-back 21.8 ms/frame   (a numpy memcpy)

Method: run each operation N times in one thread, then the same total work
split across two threads. Perfect GIL release halves the wall time; no release
leaves it unchanged. Measured before writing the pipelined engine, so the
effort is spent only if the platform can actually repay it.
"""
import sys
import threading
import time

import numpy as np
import vart
import xir

sys.path.insert(0, "/home/root")
sys.path.insert(0, "/home/root/hybrid_pkg")
from hw_ip_driver import ZoclBuffer, DRM_PATH_DEFAULT

XMODEL = "/home/root/lcam_hybrid.xmodel"
N = 24


def bench(fn, n, threads):
    """Run fn() n times total, spread over `threads` workers."""
    per = n // threads
    def worker():
        for _ in range(per):
            fn()
    ts = [threading.Thread(target=worker) for _ in range(threads)]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return time.perf_counter() - t0


def main():
    print("%-34s %10s %10s %8s" % ("operation", "1 thread", "2 threads", "gain"))
    print("-" * 66)

    # ---- 1. the write-combining read-back -------------------------------
    b = ZoclBuffer(DRM_PATH_DEFAULT, 2 << 20)
    view = np.frombuffer(b._mm, dtype=np.int8)
    n_bytes = 1638400
    view[:n_bytes] = 1

    def read_back():
        return view[:n_bytes].copy()

    t1 = bench(read_back, N, 1)
    t2 = bench(read_back, N, 2)
    print("%-34s %8.1f ms %8.1f ms %7.2fx"
          % ("numpy copy from WC mapping", t1 * 1e3 / N, t2 * 1e3 / N, t1 / t2))
    b.close()

    # ---- 2. a DPU subgraph ----------------------------------------------
    g = xir.Graph.deserialize(XMODEL)
    dpu = [s for s in g.get_root_subgraph().toposort_child_subgraph()
           if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]

    # separate runner per thread: sharing one runner across threads is not
    # safe, and a real pipelined engine would allocate per-thread runners too
    runners = [vart.Runner.create_runner(dpu[0], "run") for _ in range(2)]
    tl = threading.local()
    counter = {"i": 0}
    lock = threading.Lock()

    def dpu_run():
        if not hasattr(tl, "r"):
            with lock:
                tl.r = runners[counter["i"] % 2]
                counter["i"] += 1
            its, ots = tl.r.get_input_tensors(), tl.r.get_output_tensors()
            tl.ins = [np.zeros(tuple(t.dims), dtype=np.int8) for t in its]
            tl.outs = [np.zeros(tuple(t.dims), dtype=np.int8) for t in ots]
        tl.r.wait(tl.r.execute_async(tl.ins, tl.outs))

    t1 = bench(dpu_run, N, 1)
    t2 = bench(dpu_run, N, 2)
    print("%-34s %8.1f ms %8.1f ms %7.2fx"
          % ("DPU subgraph 0 (VART)", t1 * 1e3 / N, t2 * 1e3 / N, t1 / t2))

    # ---- 3. plain numpy, as a GIL sanity check ---------------------------
    a = np.random.rand(400, 400).astype(np.float32)

    def matmul():
        return a @ a

    t1 = bench(matmul, N, 1)
    t2 = bench(matmul, N, 2)
    print("%-34s %8.1f ms %8.1f ms %7.2fx"
          % ("numpy matmul (known to release)", t1 * 1e3 / N, t2 * 1e3 / N, t1 / t2))

    print()
    print("gain ~2.0x = fully parallel, ~1.0x = GIL-bound (threads take turns).")
    print("The DPU row is the one that decides whether pipelining is worth")
    print("building: it is 43 of the 76 ms frame.")


if __name__ == "__main__":
    main()
