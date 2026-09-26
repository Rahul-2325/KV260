#!/usr/bin/env python3
"""
test_hybrid_dpu.py -- run ON THE BOARD after loading kv260-hybrid2.

Answers the two questions that decide whether Option B actually works:

  1. Which xmodel matches this bitstream's DPU fingerprint?
     Enabling URAM has moved the fingerprint before (PROJECT_HISTORY 29.15),
     so this must be checked, never assumed.

  2. Does the DPU actually START?
     This is the exact step that failed on the Vivado-block-design hybrid:
     XRT drove the raw DPU IP with the generic AP_CTRL_HS handshake, wrote
     ap_start into what is really a read-only version register, and hung
     CU(0). Creating a runner and executing one inference is the real test.

Read-only apart from the inference itself; runs no custom-IP registers.
"""
import sys
import time
import numpy as np
import xir
import vart

XMODELS = ["/home/root/lcam_hybrid.xmodel"]


def dpu_subgraphs(graph):
    root = graph.get_root_subgraph()
    if root.is_leaf:
        return []
    out = []
    for s in root.toposort_child_subgraph():
        if s.has_attr("device") and s.get_attr("device").upper() == "DPU":
            out.append(s)
    return out


def main():
    for path in XMODELS:
        print("=" * 66)
        print(path)
        try:
            g = xir.Graph.deserialize(path)
        except Exception as e:
            print("  deserialize FAILED:", e)
            continue

        subs = dpu_subgraphs(g)
        print("  DPU subgraphs: %d" % len(subs))
        if not subs:
            print("  -> no DPU subgraph, skipping")
            continue

        s0 = subs[0]
        for attr in ("dpu_fingerprint", "fingerprint", "device_core_id"):
            if s0.has_attr(attr):
                v = s0.get_attr(attr)
                print("  %s = %s" % (attr, hex(v) if isinstance(v, int) else v))

        # THE test: build a runner. If the fingerprint does not match, VART
        # refuses here with a "fingerprint" / "target" error rather than
        # hanging -- a clean, informative failure.
        try:
            t0 = time.perf_counter()
            r = vart.Runner.create_runner(s0, "run")
            print("  create_runner OK  (%.1f ms)" % ((time.perf_counter() - t0) * 1e3))
        except Exception as e:
            print("  create_runner FAILED: %s" % e)
            continue

        it = r.get_input_tensors()
        ot = r.get_output_tensors()
        print("  inputs : %s" % [(t.name, tuple(t.dims)) for t in it])
        print("  outputs: %s" % [(t.name, tuple(t.dims)) for t in ot])

        # One real inference on zeros. If the DPU does not start, this is
        # where it would hang or time out -- which is precisely the failure
        # the Vivado-flow hybrid produced.
        ins = [np.zeros(tuple(t.dims), dtype=np.int8) for t in it]
        outs = [np.zeros(tuple(t.dims), dtype=np.int8) for t in ot]
        try:
            t0 = time.perf_counter()
            jid = r.execute_async(ins, outs)
            r.wait(jid)
            dt = (time.perf_counter() - t0) * 1e3
            print("  *** DPU EXECUTED OK: %.2f ms for subgraph 0 ***" % dt)
        except Exception as e:
            print("  execute FAILED: %s" % e)
            del r
            continue

        # timed loop, so we get a real per-subgraph number
        N = 20
        t0 = time.perf_counter()
        for _ in range(N):
            r.wait(r.execute_async(ins, outs))
        dt = (time.perf_counter() - t0) * 1e3 / N
        print("  steady-state: %.2f ms/run over %d runs" % (dt, N))
        print("  total DPU subgraphs to run per frame: %d" % len(subs))
        del r

    print("=" * 66)
    print("DONE")


if __name__ == "__main__":
    sys.exit(main())

