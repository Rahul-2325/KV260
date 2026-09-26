#!/usr/bin/env python3
"""
test_sentinel_v2.py
Same as test_sentinel.py, but adds explicit cache sync calls around
the PL write: sync_to_device() after CPU writes input (flush to DRAM
so the PL master reads fresh data), and sync_from_device() after the
IP completes (invalidate CPU cache so we read fresh data from DRAM,
not a stale cached sentinel).
"""
from hw_ip_driver import ZoclBuffer, DRM_PATH_DEFAULT
from ip_wrappers import HeadTranspose
import numpy as np

H, W, C = 4, 4, 3
total = H * W * C

in_buf  = ZoclBuffer(DRM_PATH_DEFAULT, total)
out_buf = ZoclBuffer(DRM_PATH_DEFAULT, total)

src = np.arange(total, dtype=np.int8)
in_buf.write_array(src)
in_buf.sync_to_device()          # <-- flush CPU write to DRAM

sentinel = np.full(total, 0x55, dtype=np.int8)
out_buf.write_array(sentinel)
out_buf.sync_to_device()

before = out_buf.read_array((total,), np.int8)
print(f"Output buffer BEFORE run: {before.tolist()}")

ip = HeadTranspose()
elapsed = ip.run(in_buf.phys_addr, out_buf.phys_addr, H, W, C)
ip.close()
print(f"HW call completed in {elapsed*1000:.4f} ms")

out_buf.sync_from_device()       # <-- invalidate CPU cache, force fresh read
after = out_buf.read_array((total,), np.int8)
print(f"Output buffer AFTER  run: {after.tolist()}")

expected = np.zeros((H, W, C), dtype=np.int8)
src_chw = src.reshape(C, H, W)
for c in range(C):
    for h in range(H):
        for w in range(W):
            expected[h, w, c] = src_chw[c, h, w]
expected = expected.flatten()

print(f"Expected:                 {expected.tolist()}")

if np.array_equal(after, sentinel):
    print("\n=> STILL unchanged -- not a cache issue, something else.")
elif np.array_equal(after, expected):
    print("\n=> PASS! Cache coherency was the bug. Fixed.")
else:
    print(f"\n=> Changed but not correct -- {np.sum(after==expected)}/{total} match")

in_buf.close()
out_buf.close()
