#!/usr/bin/env python3
"""
spike_zerocopy.py -- run ON THE BOARD. Proves (or kills) the zero-copy route.

The pipeline's biggest remaining cost is copying the LCAM CU's output out of
a write-combining mapping: 3.07 MB at ~140 MB/s = 21.57 ms of a 75.5 ms
frame. Allocation flags cannot fix it (§36.1 -- CMA, COHERENT and
CMA|COHERENT all read at 135 MB/s, and 2022.2 has no CACHEABLE flag).

The fix is to never copy: have the CU write its result STRAIGHT INTO the
buffer the next DPU subgraph reads from. For that we need that buffer's
PHYSICAL address, and vart.TensorBuffer exposes only get_tensor() -- no
data_phy(). So translate its virtual address ourselves via /proc/self/pagemap.

Three things must hold, and each is checked here rather than assumed:
  1. RunnerExt hands out TensorBuffers we can get a numpy view of.
  2. pagemap gives a physical address for that view.
  3. That address is REALLY the same memory -- verified by having the CU
     write a known pattern to it and reading it back through the view.

Check 3 is the one that matters. A plausible-looking address that points
somewhere else is exactly the failure mode that caused the all-zeros bug
(§26), so nothing here is believed without a data round-trip.
"""
import mmap
import os
import struct
import sys

import numpy as np
import vart
import xir

sys.path.insert(0, "/home/root")
from hw_ip_driver import ZoclBuffer, DRM_PATH_DEFAULT

XMODEL = "/home/root/lcam_hybrid.xmodel"
PAGE = 4096

CU_BASE = 0xA0020000
R_AP, R_FIN, R_WIN, R_FOUT = 0x00, 0x10, 0x1C, 0x28
R_H, R_W, R_C, R_FPF, R_FPW, R_FPO = 0x34, 0x3C, 0x44, 0x4C, 0x54, 0x5C
AP_START, AP_DONE, AP_IDLE, AP_CONTINUE = 0x1, 0x2, 0x4, 0x10


def virt_to_phys(vaddr):
    """Translate a userspace virtual address via /proc/self/pagemap (needs root)."""
    with open("/proc/self/pagemap", "rb") as f:
        f.seek((vaddr // PAGE) * 8)
        raw = f.read(8)
    if len(raw) != 8:
        raise RuntimeError("short pagemap read")
    entry = struct.unpack("<Q", raw)[0]
    if not (entry & (1 << 63)):
        raise RuntimeError("page not present (entry=0x%x)" % entry)
    pfn = entry & ((1 << 55) - 1)
    if pfn == 0:
        raise RuntimeError("PFN reads 0 -- no permission to see it")
    return pfn * PAGE + (vaddr % PAGE)


def main():
    g = xir.Graph.deserialize(XMODEL)
    subs = g.get_root_subgraph().toposort_child_subgraph()
    dpu = [s for s in subs
           if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]

    # subgraph index 3 overall = the DPU stage whose INPUT is gate 1's output
    target = dpu[1]
    print("target DPU subgraph: %s" % target.get_name()[-60:])

    print("\n--- 1. can RunnerExt give us TensorBuffers? ---")
    try:
        r = vart.RunnerExt.create_runner(target, "run")
    except Exception as e:
        print("  RunnerExt.create_runner FAILED: %s" % e)
        return 1
    try:
        tbs = r.get_inputs()
    except Exception as e:
        print("  get_inputs() FAILED: %s" % e)
        return 1
    print("  get_inputs() -> %d buffer(s)" % len(tbs))

    tb = tbs[0]
    t = tb.get_tensor()
    print("  tensor: %s  dims=%s" % (t.name[-46:], list(t.dims)))

    try:
        arr = np.asarray(tb)
    except Exception as e:
        print("  np.asarray(TensorBuffer) FAILED: %s" % e)
        return 1
    print("  numpy view: shape=%s dtype=%s  writable=%s"
          % (arr.shape, arr.dtype, arr.flags.writeable))

    vaddr = arr.__array_interface__["data"][0]
    print("  virtual addr = 0x%x" % vaddr)

    print("\n--- 2. pagemap translation ---")
    try:
        paddr = virt_to_phys(vaddr)
    except Exception as e:
        print("  FAILED: %s" % e)
        return 1
    print("  physical addr = 0x%x   %s"
          % (paddr, "LOW-DDR (PL can reach)" if paddr < 0x80000000
             else "HIGH DDR -- PL CANNOT REACH (§26)"))

    # contiguity: DPU buffers are CMA, but verify rather than assume
    n = int(np.prod(arr.shape))
    pages = min(8, n // PAGE)
    contig = True
    for k in range(1, pages):
        if virt_to_phys(vaddr + k * PAGE) != paddr + k * PAGE:
            contig = False
            break
    print("  first %d pages physically contiguous: %s" % (pages, contig))

    print("\n--- 3. THE REAL TEST: make the CU write there, read it back ---")
    H, W, C = t.dims[1], t.dims[2], t.dims[3]
    nf = H * W * C
    print("  shape %dx%dx%d = %d bytes" % (H, W, C, nf))

    bi = ZoclBuffer(DRM_PATH_DEFAULT, 2 << 20)
    bw = ZoclBuffer(DRM_PATH_DEFAULT, 2 << 20)
    vi = np.frombuffer(bi._mm, dtype=np.int8)
    vw = np.frombuffer(bw._mm, dtype=np.int8)

    # feat = 1, gate = 32, fp 5/5/5 -> out = 1*32 >> 5 = 1 everywhere
    vi[:nf] = 1
    bi.sync_to_device(size=nf)
    vw[:H * W] = 32
    bw.sync_to_device(size=H * W)

    arr[...] = 0                      # poison, so a stale read is obvious

    fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
    mm = mmap.mmap(fd, 0x1000, mmap.MAP_SHARED,
                   mmap.PROT_READ | mmap.PROT_WRITE, offset=CU_BASE)

    def w32(off, v):
        struct.pack_into("<I", mm, off, v & 0xFFFFFFFF)

    def r32(off):
        return struct.unpack_from("<I", mm, off)[0]

    def w64(off, v):
        w32(off, v & 0xFFFFFFFF)
        w32(off + 4, (v >> 32) & 0xFFFFFFFF)

    if not (r32(R_AP) & AP_IDLE):
        print("  CU not idle (0x%x) -- reload the app first" % r32(R_AP))
        return 1

    w64(R_FIN, bi.phys_addr)
    w64(R_WIN, bw.phys_addr)
    w64(R_FOUT, paddr)                # <-- straight into VART's buffer
    for off, val in ((R_H, H), (R_W, W), (R_C, C),
                     (R_FPF, 5), (R_FPW, 5), (R_FPO, 5)):
        w32(off, val)
    w32(R_AP, AP_START)

    import time
    t0 = time.perf_counter()
    while not (r32(R_AP) & AP_DONE):
        if time.perf_counter() - t0 > 5:
            print("  CU timeout")
            return 1
    w32(R_AP, AP_CONTINUE)
    print("  CU completed in %.2f ms" % ((time.perf_counter() - t0) * 1e3))

    got = np.asarray(tb).reshape(-1)[:nf]
    ones = int((got == 1).sum())
    print("  values == 1 : %d of %d" % (ones, nf))
    print("  first 16    : %s" % got[:16].tolist())
    print()
    if ones == nf:
        print("  *** ZERO-COPY WORKS -- the CU wrote directly into VART's buffer.")
        print("      This removes the 21.57 ms read-back entirely.")
    elif ones == 0:
        print("  *** FAILED: buffer untouched. The physical address is wrong,")
        print("      or VART's buffer is not where pagemap says it is.")
    else:
        print("  *** PARTIAL (%d/%d): buffer is probably not contiguous." % (ones, nf))

    mm.close(); os.close(fd)
    bi.close(); bw.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
