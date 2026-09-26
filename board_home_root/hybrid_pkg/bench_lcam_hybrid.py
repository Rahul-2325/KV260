#!/usr/bin/env python3
"""
bench_lcam_hybrid.py -- run ON THE BOARD with kv260-hybrid2 loaded.

Measures the optimised LCAM attention gate AS IT EXISTS IN THE HYBRID
BITSTREAM, alongside the DPU, and verifies it is still bit-exact.

WHY A NEW SCRIPT rather than reusing bench_lcam_opt.py:

  1. NEW BASE ADDRESS. v++ assigned this CU 0xa0020000, not the old
     0x80060000. Poking the old address would now hit the DPU's aperture.

  2. NEW REGISTER MAP. Making this a legal Vitis kernel required collapsing
     the split 'control'/'control_r' AXI-Lite bundles into one (see
     PROJECT_HISTORY 31.3), so every argument offset moved. Taken verbatim
     from the generated xlcam_attention_gate_opt_hw.h:
        0x00 ap_ctrl   0x10 feat_in(64b)  0x1c weight_in(64b)
        0x28 feat_out(64b)
        0x34 H  0x3c W  0x44 C  0x4c fp_feat  0x54 fp_weight  0x5c fp_out

  3. NEW BUS WIDTH. 128-bit instead of 512-bit, to match the real S_AXI_HP
     port width (31.4). Semantics are unchanged, so the reference model
     below is the same one proven bit-identical to the original float IP.

Buffers still come from ZoclBuffer, which carries the MIN_CMA_ALLOC fix --
allocations under 64 KB land in HIGH DDR that the PL simply cannot reach,
which was the all-zeros root cause (section 26). The assert below re-checks it.
"""
import mmap
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, "/home/root")
from hw_ip_driver import ZoclBuffer, DRM_PATH_DEFAULT

CU_BASE = 0xA0020000          # from xbutil examine / xclbinutil IP_LAYOUT
CU_SIZE = 0x1000

# CONTROL PROTOCOL: ap_ctrl_chain, NOT ap_ctrl_hs.
#
# This is the one behavioural difference between a v++-linked kernel and the
# nine IP-flow cores, and it is easy to miss because the register offsets
# look familiar. From the generated xlcam_attention_gate_opt_hw.h:
#     bit 0 - ap_start    (Read/Write/COH)
#     bit 1 - ap_done     (Read)              <-- sticky, NOT clear-on-read
#     bit 2 - ap_idle     (Read)
#     bit 4 - ap_continue (Read/Write/SC)
#
# The 9-IP driver's poll_ap_done() assumes ap_ctrl_hs, where reading ap_done
# clears it. Here it does not: ap_done latches until ap_continue is written.
# Reusing that loop makes the FIRST run look correct and then every later run
# return in ~30 us against an untouched output buffer, because the poll sees
# the previous run's done bit still asserted. Observed exactly that on the
# first attempt: layer 23 bit-exact at 1.21 ms, layers 41/53/83 "instant" and
# fully mismatched, CU finally stuck at ap_ctrl=0x203.
AP_START    = 0x01
AP_DONE     = 0x02
AP_IDLE     = 0x04
AP_CONTINUE = 0x10

R_AP_CTRL   = 0x00
R_FEAT_IN   = 0x10
R_WEIGHT_IN = 0x1C
R_FEAT_OUT  = 0x28
R_H         = 0x34
R_W         = 0x3C
R_C         = 0x44
R_FP_FEAT   = 0x4C
R_FP_WEIGHT = 0x54
R_FP_OUT    = 0x5C

BUF = 8 << 20

# the four real gate layers: (id, H, W, C, fp_feat, fp_weight, fp_out)
LAYERS = [
    (23, 160, 160,  64, 4, 7, 5),
    (41,  80,  80, 128, 5, 7, 5),
    (53,  40,  40, 256, 5, 7, 5),
    (83,  20,  20, 512, 5, 7, 6),
]


class CU(object):
    """Minimal AXI-Lite access to the kernel's single control bundle."""

    def __init__(self, base=CU_BASE):
        self.fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, CU_SIZE, mmap.MAP_SHARED,
                            mmap.PROT_READ | mmap.PROT_WRITE, offset=base)

    def w32(self, off, val):
        struct.pack_into("<I", self.mm, off, val & 0xFFFFFFFF)

    def r32(self, off):
        return struct.unpack_from("<I", self.mm, off)[0]

    def w64(self, off, val):
        self.w32(off, val & 0xFFFFFFFF)
        self.w32(off + 4, (val >> 32) & 0xFFFFFFFF)

    def wait_idle(self, timeout=10.0):
        """
        Block until ap_idle (bit 2) is high.

        This is NOT optional. ap_done (bit 1) in the AXI-Lite control block
        is clear-on-read, so a poll loop that breaks the instant it sees
        ap_done can leave the *next* invocation reading a stale done bit and
        returning immediately -- the CU never starts, the output buffer keeps
        whatever was in it, and the run "completes" in ~30 us. That is
        exactly what happened to layers 41/53/83 on the first attempt:
        0.03 ms each and a full-tensor mismatch against a 0x55 prefill.
        Gating every launch on ap_idle removes the ambiguity.
        """
        t0 = time.perf_counter()
        while True:
            s = self.r32(R_AP_CTRL)
            if s & AP_IDLE:
                return s
            if time.perf_counter() - t0 > timeout:
                raise RuntimeError("CU never went idle, ap_ctrl=0x%x" % s)

    def run(self, a_in, a_w, a_out, H, W, C, fpf, fpw, fpo, timeout=10.0):
        self.wait_idle()
        self.w64(R_FEAT_IN, a_in)
        self.w64(R_WEIGHT_IN, a_w)
        self.w64(R_FEAT_OUT, a_out)
        self.w32(R_H, H)
        self.w32(R_W, W)
        self.w32(R_C, C)
        self.w32(R_FP_FEAT, fpf)
        self.w32(R_FP_WEIGHT, fpw)
        self.w32(R_FP_OUT, fpo)

        t0 = time.perf_counter()
        self.w32(R_AP_CTRL, AP_START)
        while True:
            s = self.r32(R_AP_CTRL)
            if s & AP_DONE:
                break
            if time.perf_counter() - t0 > timeout:
                raise RuntimeError("CU timeout, ap_ctrl=0x%x" % s)
        dt = time.perf_counter() - t0

        # MANDATORY for ap_ctrl_chain: acknowledge the done.
        self.w32(R_AP_CTRL, AP_CONTINUE)
        return dt

    def close(self):
        self.mm.close()
        os.close(self.fd)


def reference(feat, gate_pix, fpf, fpw, fpo, H, W):
    """Exact integer model, half-away-from-zero, as the IP implements it."""
    shift = fpf + fpw - fpo
    prod = feat.astype(np.int32) * gate_pix.reshape(H, W, 1).astype(np.int32)
    bias = 1 << (shift - 1)
    mag = np.abs(prod)
    r = (mag + bias) >> shift
    return np.clip(np.where(prod < 0, -r, r), -128, 127).astype(np.int8)


def main():
    cu = CU()
    print("CU ap_ctrl @0x%08x = 0x%x  (0x4 = IDLE)" % (CU_BASE, cu.r32(R_AP_CTRL)))

    bi = ZoclBuffer(DRM_PATH_DEFAULT, BUF)
    bw = ZoclBuffer(DRM_PATH_DEFAULT, BUF)
    bo = ZoclBuffer(DRM_PATH_DEFAULT, BUF)
    print("buffers: in=0x%x w=0x%x out=0x%x" % (bi.phys_addr, bw.phys_addr, bo.phys_addr))
    for b in (bi, bw, bo):
        assert b.phys_addr < 0x80000000, "buffer in HIGH DDR - PL cannot reach it"
    print("all buffers in LOW DDR (PL-reachable)\n")

    rng = np.random.default_rng(0)
    print("%-5s %-16s %10s %10s %9s  %s"
          % ("layer", "shape", "HW (ms)", "CPU (ms)", "speedup", "correct"))
    print("-" * 74)

    tot_hw = tot_cpu = 0.0
    all_ok = True
    for lid, H, W, C, fpf, fpw, fpo in LAYERS:
        feat = rng.integers(-128, 128, (H, W, C), dtype=np.int8)
        gate_pix = rng.integers(-128, 128, (H * W,), dtype=np.int8)

        bi.write_array(feat); bi.sync_to_device()
        bw.write_array(gate_pix); bw.sync_to_device()
        bo.write_array(np.full(H * W * C, 0x55, dtype=np.int8)); bo.sync_to_device()

        t = cu.run(bi.phys_addr, bw.phys_addr, bo.phys_addr, H, W, C, fpf, fpw, fpo)
        bo.sync_from_device()
        got = bo.read_array((H, W, C), np.int8)

        t0 = time.perf_counter()
        exp = reference(feat, gate_pix, fpf, fpw, fpo, H, W)
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
    print()
    print("reference points from PROJECT_HISTORY:")
    print("  original IP (512-bit, 100 MHz standalone) : 40.60 ms")
    print("  optimised IP (512-bit, 100 MHz standalone):  3.50 ms")
    print("  numpy int8 exact-rounding CPU             : 137.97 ms")
    print("  real VART CPU fallback for the gates      :  83.20 ms")

    cu.close()
    for b in (bi, bw, bo):
        b.close()


if __name__ == "__main__":
    main()

