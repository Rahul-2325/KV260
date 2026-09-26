#!/usr/bin/env python3
"""
measure_gate_path_cpu.py -- run ON THE BOARD.

Question: of the 385.36 ms the CPU spends per frame, how much is the LCAM
attention-gate path that our IP actually replaces?

`enumerate_cpu_ops.py` gives the composition by element count:
    fix2float  18 ops  3,248,800 elem
    float2fix  13 ops  3,156,000 elem
    mul         4 ops  3,072,000 elem     <- the four gates
    everything else    ~378,000 elem
The gate feature tensors total exactly 3,072,000 elements, and the
fix2float / float2fix totals sit right on top of that, which says the
conversions exist BECAUSE the multiply is done in float on the CPU.

Element counts are a proxy, not time, so this script TIMES the real chain
VART executes for one gate:
      int8 -> float32  (fix2float, both operands)
      float32 multiply (broadcast the per-pixel gate over channels)
      float32 -> int8  (float2fix, round + saturate)
at the four true layer sizes. If that lands near 385 ms, the attribution
is confirmed and the IP's 1.82 ms replaces essentially all of it.

Our IP does the whole chain in the int8 domain, so it removes the
conversions as well as the multiply.
"""
import time

import numpy as np

# (id, H, W, C, fp_feat, fp_weight, fp_out)
LAYERS = [
    (23, 160, 160,  64, 4, 7, 5),
    (41,  80,  80, 128, 5, 7, 5),
    (53,  40,  40, 256, 5, 7, 5),
    (83,  20,  20, 512, 5, 7, 6),
]

REPS = 5


def gate_path_float(feat_i8, gate_i8, fpf, fpw, fpo, H, W):
    """Exactly what VART does on the CPU: dequantise, multiply, requantise."""
    f = feat_i8.astype(np.float32) * np.float32(2.0 ** -fpf)      # fix2float
    g = gate_i8.astype(np.float32) * np.float32(2.0 ** -fpw)      # fix2float
    p = f * g.reshape(H, W, 1)                                    # mul
    q = np.round(p * np.float32(2.0 ** fpo))                      # float2fix
    return np.clip(q, -128, 127).astype(np.int8)


def main():
    rng = np.random.default_rng(0)
    print("%-5s %-16s %12s %12s" % ("layer", "shape", "elements", "CPU ms"))
    print("-" * 50)

    total = 0.0
    total_elems = 0
    for lid, H, W, C, fpf, fpw, fpo in LAYERS:
        feat = rng.integers(-128, 128, (H, W, C), dtype=np.int8)
        gate = rng.integers(-128, 128, (H * W,), dtype=np.int8)

        gate_path_float(feat, gate, fpf, fpw, fpo, H, W)          # warm

        t0 = time.perf_counter()
        for _ in range(REPS):
            gate_path_float(feat, gate, fpf, fpw, fpo, H, W)
        dt = (time.perf_counter() - t0) / REPS * 1e3

        total += dt
        total_elems += H * W * C
        print("%-5d %-16s %12s %12.2f"
              % (lid, "%dx%dx%d" % (H, W, C), "{:,}".format(H * W * C), dt))

    print("-" * 50)
    print("%-5s %-16s %12s %12.2f" % ("TOTAL", "", "{:,}".format(total_elems), total))
    print()
    print("measured on this board, for reference:")
    print("  end-to-end (DPU + CPU fallback)  : 426.51 ms   2.34 FPS")
    print("  DPU compute (8 subgraphs)        :  41.15 ms")
    print("  CPU fallback (by subtraction)    : 385.36 ms")
    print("  custom LCAM IP, all four gates   :   1.82 ms   (bit-exact)")
    print()
    print("gate path as a share of CPU fallback: %.1f%%" % (100.0 * total / 385.36))


if __name__ == "__main__":
    main()
