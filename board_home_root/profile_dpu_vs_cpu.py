#!/usr/bin/env python3
"""
profile_dpu_vs_cpu.py  -- PHASE 1b: where does the 430 ms actually go?

Times each DPU subgraph individually with vart.Runner, sums them, and
derives the CPU-fallback time as (end-to-end - DPU total).

This is THE key measurement for the paper: it quantifies how much of the
frame time is spent on the LCAM attention that the DPU cannot execute and
therefore falls back to the ARM CPU -- i.e. the headroom our custom
lcam_attention_gate IP is targeting.

DPU execution time is data-independent, so feeding zeros of the correct
shape gives valid timings.

Usage:
  export XLNX_VART_FIRMWARE=/lib/firmware/xilinx/kv260-benchmark-b4096/kv260-benchmark-b4096.xclbin
  python3 profile_dpu_vs_cpu.py --xmodel /home/root/lcam_v5_2p5.xmodel --runs 20
"""
import argparse, time
import numpy as np
import xir, vart


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xmodel', default='/home/root/lcam_v5_2p5.xmodel')
    ap.add_argument('--runs', type=int, default=20)
    ap.add_argument('--e2e_ms', type=float, default=None,
                    help='measured end-to-end latency to subtract against')
    args = ap.parse_args()

    g = xir.Graph.deserialize(args.xmodel)
    subs = g.get_root_subgraph().toposort_child_subgraph()
    dpu_subs = [s for s in subs
                if s.has_attr('device') and s.get_attr('device') == 'DPU']
    cpu_subs = [s for s in subs
                if s.has_attr('device') and s.get_attr('device') == 'CPU']

    print("=" * 70)
    print("PHASE 1b: DPU vs CPU TIME BREAKDOWN")
    print("=" * 70)
    print("subgraphs: %d DPU, %d CPU\n" % (len(dpu_subs), len(cpu_subs)))

    total_dpu = 0.0
    print("%-4s %-34s %-10s %s" % ("#", "output shape", "mean ms", "ops"))
    print("-" * 70)
    for i, s in enumerate(dpu_subs):
        try:
            r = vart.Runner.create_runner(s, "run")
            its, ots = r.get_input_tensors(), r.get_output_tensors()
            ins = [np.zeros(list(t.dims), dtype=np.int8) for t in its]
            outs = [np.zeros(list(t.dims), dtype=np.int8) for t in ots]
            # warm
            for _ in range(3):
                jid = r.execute_async(ins, outs); r.wait(jid)
            ts = []
            for _ in range(args.runs):
                t0 = time.perf_counter()
                jid = r.execute_async(ins, outs); r.wait(jid)
                ts.append(time.perf_counter() - t0)
            ms = np.mean(ts) * 1000
            total_dpu += ms
            nops = len(s.get_children()) if s.get_children() else 0
            print("%-4d %-34s %-10.3f %d" % (i, str(list(ots[0].dims)), ms, nops))
            del r
        except Exception as e:
            print("%-4d  FAILED: %s" % (i, str(e)[:52]))

    print("-" * 70)
    print("TOTAL DPU COMPUTE : %8.2f ms" % total_dpu)
    if args.e2e_ms:
        cpu = args.e2e_ms - total_dpu
        print("END-TO-END        : %8.2f ms" % args.e2e_ms)
        print("=> CPU FALLBACK   : %8.2f ms   (%.1f%% of frame time)"
              % (cpu, 100.0 * cpu / args.e2e_ms))
        print()
        print("=" * 70)
        print("PROJECTION: replacing CPU attention with the custom IP")
        print("=" * 70)
        print("  measured custom lcam_attention_gate total = 40.6 ms")
        print("  (four gates: 23.6 + 10.1 + 4.7 + 2.2 ms, see PROJECT_HISTORY §28)")
        best = total_dpu + 40.6
        print("  projected hybrid latency = %.2f (DPU) + 40.6 (IP) = %.2f ms"
              % (total_dpu, best))
        print("  projected hybrid FPS     = %.2f   (vs %.2f now)"
              % (1000.0 / best, 1000.0 / args.e2e_ms))
        print("  projected SPEEDUP        = %.2fx" % (args.e2e_ms / best))
        print("  NOTE: assumes ALL CPU time is the attention gates. Some of")
        print("        the 15 CPU subgraphs are pooling/concat/reshape that")
        print("        the custom IP does NOT replace -- so treat this as an")
        print("        UPPER BOUND until the hybrid is actually built.")
    print("=" * 70)


if __name__ == '__main__':
    main()
