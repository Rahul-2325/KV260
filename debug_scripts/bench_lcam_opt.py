#!/usr/bin/env python3
"""
bench_lcam_opt.py
Benchmarks the OPTIMISED lcam_attention_gate (512-bit AXI, integer
arithmetic) against the CPU, and -- just as important -- VERIFIES it
produces exactly the reference result.

Register map is identical to the original IP (confirmed from the
generated xlcam_attention_gate_opt_hw.h), so ip_wrappers.LcamAttentionGate
drives it unchanged.

Reference model is the same integer formula proven bit-identical to the
original float IP over the full int8 x int8 domain
(debug_scripts/verify_lcam_int_math.py).
"""
import time
import numpy as np
from hw_ip_driver import ZoclBuffer, DRM_PATH_DEFAULT
from ip_wrappers import LcamAttentionGate

BUF = 8 << 20

# the four real gate layers: (id, H, W, C, fp_feat, fp_weight, fp_out)
LAYERS = [
    (23, 160, 160,  64, 4, 7, 5),
    (41,  80,  80, 128, 5, 7, 5),
    (53,  40,  40, 256, 5, 7, 5),
    (83,  20,  20, 512, 5, 7, 6),
]


def reference(feat, gate, fp_feat, fp_weight, fp_out):
    """exact integer reference, matching the IP"""
    shift = fp_feat + fp_weight - fp_out
    prod = feat.astype(np.int32) * gate.astype(np.int32)[None, None, :]
    if shift > 0:
        bias = 1 << (shift - 1)
        mag = np.abs(prod)
        r = (mag + bias) >> shift
        q = np.where(prod < 0, -r, r)
    else:
        q = prod << (-shift)
    return np.clip(q, -128, 127).astype(np.int8)


def main():
    bi = ZoclBuffer(DRM_PATH_DEFAULT, BUF)
    bw = ZoclBuffer(DRM_PATH_DEFAULT, BUF)
    bo = ZoclBuffer(DRM_PATH_DEFAULT, BUF)
    print("buffers: in=0x%x w=0x%x out=0x%x" % (bi.phys_addr, bw.phys_addr, bo.phys_addr))
    for b in (bi, bw, bo):
        assert b.phys_addr < 0x80000000, "buffer in HIGH DDR - PL cannot reach it"
    print("all buffers in LOW DDR (PL-reachable)\n")

    ip = LcamAttentionGate()
    rng = np.random.default_rng(0)

    print("%-5s %-16s %10s %10s %9s  %s"
          % ("layer", "shape", "HW (ms)", "CPU (ms)", "speedup", "correct"))
    print("-" * 74)
    tot_hw = tot_cpu = 0.0
    all_ok = True

    for lid, H, W, C, fpf, fpw, fpo in LAYERS:
        feat = rng.integers(-128, 128, (H, W, C), dtype=np.int8)
        # NOTE: the IP broadcasts ONE weight per pixel over all channels.
        # weight_in is [H*W]; build the reference with the same semantics.
        gate_pix = rng.integers(-128, 128, (H * W,), dtype=np.int8)

        bi.write_array(feat); bi.sync_to_device()
        bw.write_array(gate_pix); bw.sync_to_device()
        bo.write_array(np.full(H * W * C, 0x55, dtype=np.int8)); bo.sync_to_device()

        # ---- hardware ----
        t = ip.run(bi.phys_addr, bw.phys_addr, bo.phys_addr, H, W, C, fpf, fpw, fpo)
        bo.sync_from_device()
        got = bo.read_array((H, W, C), np.int8)

        # ---- CPU reference (per-pixel weight broadcast over channels) ----
        t0 = time.perf_counter()
        shift = fpf + fpw - fpo
        prod = feat.astype(np.int32) * gate_pix.reshape(H, W, 1).astype(np.int32)
        bias = 1 << (shift - 1)
        mag = np.abs(prod)
        r = (mag + bias) >> shift
        exp = np.clip(np.where(prod < 0, -r, r), -128, 127).astype(np.int8)
        cpu = time.perf_counter() - t0

        ok = np.array_equal(got, exp)
        all_ok &= ok
        nbad = int((got != exp).sum())
        tot_hw += t * 1000
        tot_cpu += cpu * 1000
        print("%-5d %-16s %10.2f %10.2f %8.2fx  %s"
              % (lid, "%dx%dx%d" % (H, W, C), t * 1000, cpu * 1000,
                 cpu / t if t > 0 else 0,
                 "OK" if ok else "MISMATCH (%d elems)" % nbad))

    print("-" * 74)
    print("%-5s %-16s %10.2f %10.2f %8.2fx"
          % ("TOTAL", "", tot_hw, tot_cpu, tot_cpu / tot_hw if tot_hw > 0 else 0))
    print()
    print("correctness:", "ALL LAYERS EXACT" if all_ok else "*** MISMATCH ***")

    ip.close()
    for b in (bi, bw, bo):
        b.close()


if __name__ == '__main__':
    main()
