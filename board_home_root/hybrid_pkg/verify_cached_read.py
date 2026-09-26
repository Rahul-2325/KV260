#!/usr/bin/env python3
"""
verify_cached_read.py -- run ON THE BOARD.

bench_cached_map.py showed the CU's output buffer can be read at 2432 MB/s
instead of 137 MB/s by mapping the SAME physical pages a second time through
/dev/mem WITHOUT O_SYNC (a cached mapping) instead of the zocl DRM mapping
(write-combining). That is 21.57 ms -> ~2.2 ms of a 75.5 ms frame.

But a cached mapping brings a hazard the write-combining one did not:
COHERENCY. After the PL writes DRAM, the CPU may still hold stale lines for
those addresses, and a read would return the PREVIOUS frame's data. That is
a silent, intermittent wrong-answer bug -- the worst kind, and exactly the
family that cost this project weeks (§26).

The claim to test: sync_from_device() invalidates by physical range, and
ARMv8 data caches are physically tagged, so the invalidate applies to ANY
mapping of those pages, including ours.

PROTOCOL (order matters):
  * inputs and the poison are written through the DRM (write-combining)
    mapping only. NEVER dirty the cached mapping for the output buffer --
    a dirty line evicted after the PL writes would clobber real data.
  * read the result only through the cached mapping, after sync_from_device.

Runs many iterations with DIFFERENT data each time, because a stale-cache
bug shows up as "iteration k returns iteration k-1's answer" and would pass
a single-shot test.
"""
import mmap
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, "/home/root")
from hw_ip_driver import ZoclBuffer, DRM_PATH_DEFAULT

CU_BASE = 0xA0020000
R_AP, R_FIN, R_WIN, R_FOUT = 0x00, 0x10, 0x1C, 0x28
R_H, R_W, R_C, R_FPF, R_FPW, R_FPO = 0x34, 0x3C, 0x44, 0x4C, 0x54, 0x5C
AP_START, AP_DONE, AP_IDLE, AP_CONTINUE = 0x1, 0x2, 0x4, 0x10

LAYERS = [(160, 160, 64, 4, 7, 5), (80, 80, 128, 5, 7, 5),
          (40, 40, 256, 5, 7, 5), (20, 20, 512, 5, 7, 6)]
ITERS = 12


def reference(feat, gate, H, W, fpf, fpw, fpo):
    shift = fpf + fpw - fpo
    prod = feat.astype(np.int32) * gate.reshape(H, W, 1).astype(np.int32)
    bias = 1 << (shift - 1)
    r = (np.abs(prod) + bias) >> shift
    return np.clip(np.where(prod < 0, -r, r), -128, 127).astype(np.int8)


def main():
    bi = ZoclBuffer(DRM_PATH_DEFAULT, 2 << 20)
    bw = ZoclBuffer(DRM_PATH_DEFAULT, 2 << 20)
    bo = ZoclBuffer(DRM_PATH_DEFAULT, 2 << 20)
    for b in (bi, bw, bo):
        if b.phys_addr >= 0x80000000:
            print("HIGH DDR, aborting")
            return 1
    vi = np.frombuffer(bi._mm, dtype=np.int8)
    vw = np.frombuffer(bw._mm, dtype=np.int8)
    vo_wc = np.frombuffer(bo._mm, dtype=np.int8)      # write-combining view

    # second, CACHED mapping of the same physical pages
    cfd = os.open("/dev/mem", os.O_RDWR)               # no O_SYNC
    cmm = mmap.mmap(cfd, bo.size, mmap.MAP_SHARED,
                    mmap.PROT_READ | mmap.PROT_WRITE, offset=bo.phys_addr)
    vo_cached = np.frombuffer(cmm, dtype=np.int8)
    print("output buffer phys=0x%x  cached view mapped\n" % bo.phys_addr)

    fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
    mm = mmap.mmap(fd, 0x1000, mmap.MAP_SHARED,
                   mmap.PROT_READ | mmap.PROT_WRITE, offset=CU_BASE)
    w32 = lambda o, v: struct.pack_into("<I", mm, o, v & 0xFFFFFFFF)
    r32 = lambda o: struct.unpack_from("<I", mm, o)[0]

    def w64(o, v):
        w32(o, v & 0xFFFFFFFF)
        w32(o + 4, (v >> 32) & 0xFFFFFFFF)

    rng = np.random.default_rng(0)
    bad_wc = bad_cached = 0
    t_wc = t_cached = 0.0

    for it in range(ITERS):
        H, W, C, fpf, fpw, fpo = LAYERS[it % len(LAYERS)]
        nf, nw = H * W * C, H * W
        feat = rng.integers(-128, 128, (H, W, C), dtype=np.int8)
        gate = rng.integers(-128, 128, (nw,), dtype=np.int8)
        exp = reference(feat, gate, H, W, fpf, fpw, fpo)

        vi[:nf] = feat.reshape(-1)
        bi.sync_to_device(size=nf)
        vw[:nw] = gate
        bw.sync_to_device(size=nw)

        # poison through the WRITE-COMBINING view only, then push to DRAM.
        # Poisoning through the cached view would leave dirty lines that could
        # be evicted over the PL's result later.
        vo_wc[:nf] = 0x55
        bo.sync_to_device(size=nf)

        while not (r32(R_AP) & AP_IDLE):
            pass
        w64(R_FIN, bi.phys_addr); w64(R_WIN, bw.phys_addr)
        w64(R_FOUT, bo.phys_addr)
        for o, v in ((R_H, H), (R_W, W), (R_C, C),
                     (R_FPF, fpf), (R_FPW, fpw), (R_FPO, fpo)):
            w32(o, v)
        w32(R_AP, AP_START)
        while not (r32(R_AP) & AP_DONE):
            pass
        w32(R_AP, AP_CONTINUE)

        bo.sync_from_device(size=nf)

        t0 = time.perf_counter()
        got_c = vo_cached[:nf].reshape(H, W, C).copy()
        t_cached += time.perf_counter() - t0

        t0 = time.perf_counter()
        got_w = vo_wc[:nf].reshape(H, W, C).copy()
        t_wc += time.perf_counter() - t0

        okc = np.array_equal(got_c, exp)
        okw = np.array_equal(got_w, exp)
        bad_cached += (not okc)
        bad_wc += (not okw)
        print("  it %2d  %3dx%3dx%-3d  cached %-9s  write-combining %-9s"
              % (it, H, W, C, "OK" if okc else "MISMATCH",
                 "OK" if okw else "MISMATCH"))

    print()
    print("cached-view reads   : %d/%d correct   %.2f ms total"
          % (ITERS - bad_cached, ITERS, t_cached * 1e3))
    print("write-combining     : %d/%d correct   %.2f ms total"
          % (ITERS - bad_wc, ITERS, t_wc * 1e3))
    print()
    if bad_cached == 0:
        print("*** CACHED READ IS COHERENT. sync_from_device() invalidates our")
        print("    mapping too. Safe to use in the pipeline: ~%.1fx faster."
              % (t_wc / t_cached if t_cached else 0))
    else:
        print("*** STALE DATA on %d iteration(s) -- sync_from_device does NOT")
        print("    cover this mapping. DO NOT use it." % bad_cached)

    cmm.close(); os.close(cfd)
    mm.close(); os.close(fd)
    for b in (bi, bw, bo):
        b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
