#!/usr/bin/env python3
"""
bench_cached_map.py -- run ON THE BOARD.

Can we read the LCAM CU's output faster than 140 MB/s?

Context: the CU's result must come back to the CPU (zero-copy is dead --
VART's TensorBuffers are pageable host heap, not CMA, see spike_zerocopy.py).
The zocl DRM mapping is WRITE-COMBINING: writes 1.4-2.6 GB/s, reads ~140 MB/s.
That read is 21.57 ms of a 75.5 ms frame. Allocation flags do not change it
(§36.1).

Idea under test: the buffer is ordinary DDR. The slowness is a property of
THE MAPPING, not the memory. Mapping the same physical pages a second time
through /dev/mem WITHOUT O_SYNC may give a cached mapping, and cached reads
should run at GB/s.

Cache coherency: ARM cache maintenance is by physical address, and the DRM
sync_from_device ioctl invalidates by physical range, so it should apply to
any mapping of those pages. This script VERIFIES the data matches rather
than assuming it.

SAFE: only ever touches this script's own ZoclBuffer. No PL DMA at all.
"""
import mmap
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/home/root")
from hw_ip_driver import ZoclBuffer, DRM_PATH_DEFAULT

N = 1638400          # largest gate tensor: 160*160*64
REPS = 10


def timed_read(view, n, label, expect):
    view[:n].copy()                                  # warm
    t0 = time.perf_counter()
    for _ in range(REPS):
        out = view[:n].copy()
    dt = (time.perf_counter() - t0) / REPS
    ok = np.array_equal(out, expect)
    print("  %-34s %7.2f ms  %7.0f MB/s   data %s"
          % (label, dt * 1e3, n / dt / 1e6, "OK" if ok else "*** WRONG ***"))
    return dt


def main():
    b = ZoclBuffer(DRM_PATH_DEFAULT, 2 << 20)
    print("buffer phys=0x%x  size=%d" % (b.phys_addr, b.size))
    if b.phys_addr >= 0x80000000:
        print("HIGH DDR -- unexpected; aborting")
        return 1

    pattern = (np.arange(N, dtype=np.int32) % 127).astype(np.int8)
    drm_view = np.frombuffer(b._mm, dtype=np.int8)
    drm_view[:N] = pattern
    b.sync_to_device(size=N)

    print("\nread bandwidth, same physical pages, different mappings:")
    base = timed_read(drm_view, N, "zocl DRM mmap (current)", pattern)

    # /dev/mem WITHOUT O_SYNC -> may be a cached mapping
    for flags, label in ((os.O_RDWR, "/dev/mem  no O_SYNC (cached?)"),
                         (os.O_RDWR | os.O_SYNC, "/dev/mem  with O_SYNC")):
        fd = mm = None
        try:
            fd = os.open("/dev/mem", flags)
            mm = mmap.mmap(fd, b.size, mmap.MAP_SHARED,
                           mmap.PROT_READ | mmap.PROT_WRITE,
                           offset=b.phys_addr)
            v = np.frombuffer(mm, dtype=np.int8)
            timed_read(v, N, label, pattern)
        except Exception as e:
            print("  %-34s FAILED: %s" % (label, e))
        finally:
            if mm is not None:
                mm.close()
            if fd is not None:
                os.close(fd)

    # is numpy's copy the problem, or the mapping? try other read paths
    print("\nother read paths through the DRM mapping:")
    t0 = time.perf_counter()
    for _ in range(REPS):
        raw = b._mm[:N]
    dt = (time.perf_counter() - t0) / REPS
    print("  %-34s %7.2f ms  %7.0f MB/s   data %s"
          % ("mmap slice -> bytes", dt * 1e3, N / dt / 1e6,
             "OK" if raw == pattern.tobytes() else "*** WRONG ***"))

    t0 = time.perf_counter()
    for _ in range(REPS):
        out = np.empty(N, dtype=np.int8)
        np.copyto(out, drm_view[:N])
    dt = (time.perf_counter() - t0) / REPS
    print("  %-34s %7.2f ms  %7.0f MB/s   data %s"
          % ("np.copyto into preallocated", dt * 1e3, N / dt / 1e6,
             "OK" if np.array_equal(out, pattern) else "*** WRONG ***"))

    print()
    print("frame maths: the four gates move 3.07 MB out per frame.")
    print("  at %6.0f MB/s (today) -> %5.2f ms" % (N / base / 1e6, 3.07e6 / (N / base) * 1e3))
    print("  at 1400 MB/s           ->  2.19 ms   (frame 75.5 -> ~56 ms, ~17.8 FPS)")
    b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
