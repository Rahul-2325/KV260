#!/usr/bin/env python3
"""
measure_head_numpy.py -- run ON THE BOARD.

Decides whether the YOLOX head needs custom IPs at all.

Measured with DEEPHI_PROFILING, VART's CPU runner spends ~193 ms/frame on
the head: 91.23 ms on the final [1,8400,7] fix/transpose/fix2float, and
72.48 ms on an 80x80x7 transpose. Those tensors are ~50,000 elements. That
is not a plausible amount of work -- numpy should do each in well under a
millisecond -- which suggests the cost is VART's per-op overhead, not the
computation.

This matters for planning. A hand-written pipeline replaces VART's CPU ops
with numpy anyway. If numpy is ~100x faster here, then moving the head to
head_transpose / head_concat_reshape / head_sigmoid buys almost nothing,
and the whole bitstream rebuild for them can be skipped.

Shapes below are taken from enumerate_cpu_ops.py on lcam_hybrid.xmodel.
"""
import time

import numpy as np

REPS = 20


def timeit(fn, *a):
    fn(*a)                                   # warm
    t0 = time.perf_counter()
    for _ in range(REPS):
        fn(*a)
    return (time.perf_counter() - t0) / REPS * 1e3


def sink_transpose(x_i8, fp):
    """fix2float -> transpose NHWC->NCHW -> float2fix, as VART does it."""
    f = x_i8.astype(np.float32) * np.float32(2.0 ** -fp)
    t = np.ascontiguousarray(np.transpose(f, (0, 3, 1, 2)))
    return np.clip(np.round(t * np.float32(2.0 ** fp)), -128, 127).astype(np.int8)


def final_output(x_i8, fp):
    """fix -> transpose -> fix2float on the [1,8400,7] head output."""
    t = np.ascontiguousarray(np.transpose(x_i8, (0, 2, 1)))
    return t.astype(np.float32) * np.float32(2.0 ** -fp)


def concat_reshape(a, b, c):
    ra = a.reshape(1, 1, 7, -1)
    rb = b.reshape(1, 1, 7, -1)
    rc = c.reshape(1, 1, 7, -1)
    cat = np.concatenate([ra, rb, rc], axis=3)
    return cat.reshape(1, 7, -1)


def sigmoid_i8(x_i8, fp):
    f = x_i8.astype(np.float32) * np.float32(2.0 ** -fp)
    s = 1.0 / (1.0 + np.exp(-f))
    return np.clip(np.round(s * np.float32(2.0 ** fp)), -128, 127).astype(np.int8)


def main():
    rng = np.random.default_rng(0)
    rows = []

    for hw, vart_ms in ((20, 1.17), (40, 4.56), (80, 18.12)):
        x = rng.integers(-128, 128, (1, hw, hw, 7), dtype=np.int8)
        rows.append(("sink_transpose %dx%dx7" % (hw, hw), hw * hw * 7,
                     timeit(sink_transpose, x, 4), vart_ms))

    a = rng.integers(-128, 128, (1, 7, 20, 20), dtype=np.int8)
    b = rng.integers(-128, 128, (1, 7, 40, 40), dtype=np.int8)
    c = rng.integers(-128, 128, (1, 7, 80, 80), dtype=np.int8)
    rows.append(("concat+reshape -> [1,7,8400]", 58800,
                 timeit(concat_reshape, a, b, c), 0.10))

    out = rng.integers(-128, 128, (1, 7, 8400), dtype=np.int8)
    rows.append(("final [1,8400,7] fix/transpose/fix2float", 58800,
                 timeit(final_output, out, 4), 91.23))

    for hw, vart_ms in ((20, 0.05), (40, 0.05), (80, 0.05)):
        s = rng.integers(-128, 128, (1, hw, hw, 2), dtype=np.int8)
        rows.append(("sigmoid %dx%dx2" % (hw, hw), hw * hw * 2,
                     timeit(sigmoid_i8, s, 4), vart_ms))

    print("%-42s %10s %10s %10s %8s"
          % ("op", "elements", "numpy ms", "VART ms", "ratio"))
    print("-" * 86)
    tn = tv = 0.0
    for name, n, npms, vms in rows:
        tn += npms
        tv += vms
        print("%-42s %10s %10.3f %10.2f %8s"
              % (name, "{:,}".format(n), npms, vms,
                 ("%.0fx" % (vms / npms)) if npms > 0 else "-"))
    print("-" * 86)
    print("%-42s %10s %10.3f %10.2f" % ("TOTAL (ops listed here)", "", tn, tv))
    print()
    print("VART's measured head cost, all ops        : ~193 ms/frame")
    print("numpy equivalent of the same work         : %.2f ms" % tn)
    print()
    print("If the pipeline is hand-written in numpy, projected end-to-end:")
    e2e = 41.15 + 1.82 + tn
    print("  41.15 (DPU) + 1.82 (LCAM IP) + %.2f (numpy glue) = %.2f ms -> %.1f FPS"
          % (tn, e2e, 1000.0 / e2e))
    print()
    print("  (vs 426.51 ms / 2.34 FPS today)")
    print()
    print("NOTE: this measures the OPS only. A real pipeline also pays for")
    print("      DPU submit/wait per subgraph and any buffer copies, which")
    print("      are NOT included here. Treat as a floor, not a promise.")


if __name__ == "__main__":
    main()
