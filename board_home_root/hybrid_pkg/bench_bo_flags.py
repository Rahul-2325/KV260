#!/usr/bin/env python3
"""
bench_bo_flags.py -- run ON THE BOARD.

The pipeline's remaining bottleneck is reading the CU's output back:
3.07 MB at 143 MB/s = 21.5 ms, while WRITING the same bytes runs at
1.4 GB/s. That 10x asymmetry says the zocl mapping is write-combining.

??35.4 proposed allocating the buffer "cacheable". Checking the real
zynq_ioctl.h for XRT 2022.2 shows THERE IS NO CACHEABLE FLAG:

    DRM_ZOCL_BO_FLAGS_HOST_BO   (0x1 << 26)
    DRM_ZOCL_BO_FLAGS_COHERENT  (0x1 << 27)
    DRM_ZOCL_BO_FLAGS_CMA       (0x1 << 28)
    DRM_ZOCL_BO_FLAGS_SVM       (0x1 << 29)
    DRM_ZOCL_BO_FLAGS_USERPTR   (0x1 << 30)
    DRM_ZOCL_BO_FLAGS_EXECBUF   (0x1 << 31)

COHERENT is the only plausible candidate. This measures it in isolation
rather than changing the pipeline on a guess: allocate the same buffer
with CMA and with CMA|COHERENT, and time read and write bandwidth.

Also checks that a COHERENT buffer still lands in LOW DDR -- if it does
not, the PL cannot reach it at all (PROJECT_HISTORY section 26) and the
flag is unusable regardless of speed.
"""
import time

import numpy as np

import fcntl
import mmap
import os
import sys

sys.path.insert(0, "/home/root")
import hw_ip_driver as hd
from hw_ip_driver import DRM_PATH_DEFAULT

CMA = 0x1 << 28
COHERENT = 0x1 << 27
HOST_BO = 0x1 << 26

N = 1638400          # the largest gate tensor, 160*160*64
REPS = 10


class Bo(object):
    """
    Minimal zocl BO allocator that takes an explicit flags value.

    ZoclBuffer hardcodes CMA and takes no flags argument, and it is NOT
    edited here: it carries the MIN_CMA_ALLOC fix from section 26 and is used
    by every working script in the project. This is a local copy for the
    experiment only.
    """

    def __init__(self, size, flags):
        size = max(size, 64 * 1024)          # section 26: below 64 KB -> HIGH DDR
        self.size = size
        self._fd = os.open(DRM_PATH_DEFAULT, os.O_RDWR)
        c = hd.drm_zocl_create_bo(size=size, handle=0, flags=flags)
        fcntl.ioctl(self._fd, hd.ZOCL_IOCTL_CREATE_BO, c, True)
        self.handle = c.handle
        i = hd.drm_zocl_info_bo(handle=self.handle, pad=0, size=0, paddr=0)
        fcntl.ioctl(self._fd, hd.ZOCL_IOCTL_INFO_BO, i, True)
        self.phys_addr = i.paddr
        m = hd.drm_zocl_map_bo(handle=self.handle, pad=0, offset=0)
        fcntl.ioctl(self._fd, hd.ZOCL_IOCTL_MAP_BO, m, True)
        self._mm = mmap.mmap(self._fd, size, mmap.MAP_SHARED,
                             mmap.PROT_READ | mmap.PROT_WRITE, offset=m.offset)

    def close(self):
        self._mm.close()
        os.close(self._fd)


def bench(flags, label):
    try:
        b = Bo(2 << 20, flags)
    except Exception as e:
        print("%-22s ALLOC FAILED: %s" % (label, e))
        return

    try:
        low = b.phys_addr < 0x80000000
        view = np.frombuffer(b._mm, dtype=np.int8)
        src = np.ones(N, dtype=np.int8)

        view[:N] = src                      # warm
        t0 = time.perf_counter()
        for _ in range(REPS):
            view[:N] = src
        wt = (time.perf_counter() - t0) / REPS

        _ = view[:N].copy()                 # warm
        t0 = time.perf_counter()
        for _ in range(REPS):
            _ = view[:N].copy()
        rt = (time.perf_counter() - t0) / REPS

        print("%-22s phys=0x%09x %-8s  write %6.2f ms (%6.0f MB/s)   read %6.2f ms (%6.0f MB/s)"
              % (label, b.phys_addr, "LOW-DDR" if low else "HIGH!!",
                 wt * 1e3, N / wt / 1e6, rt * 1e3, N / rt / 1e6))
    finally:
        b.close()


bench(CMA, "CMA (current)")
bench(CMA | COHERENT, "CMA|COHERENT")
bench(COHERENT, "COHERENT only")
print()
print("baseline from the pipeline: write 1.4 GB/s, read 143 MB/s")
print("a read anywhere near write speed here means COHERENT is the fix.")

