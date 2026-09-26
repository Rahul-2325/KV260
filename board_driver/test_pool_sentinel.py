#!/usr/bin/env python3
"""
test_pool_sentinel.py
Cross-check: pool_engine sits on smartconnect_hp1, a DIFFERENT
interconnect from head_transpose (which is on smartconnect_hp2).
If pool_engine ALSO fails identically, the bug is systemic across
the whole design. If pool_engine WORKS, the bug is isolated to
smartconnect_hp2 (or head_transpose specifically) -- a much smaller,
more fixable problem.
"""
from hw_ip_driver import ZoclBuffer, DRM_PATH_DEFAULT
from ip_wrappers import PoolEngine
import numpy as np

# Small global-average-pool test: [1,4,4,2] -> [1,1,1,2]
H, W, C = 4, 4, 2
n_in = H * W * C

in_buf  = ZoclBuffer(DRM_PATH_DEFAULT, n_in)
out_buf = ZoclBuffer(DRM_PATH_DEFAULT, C)

# channel 0: all 10s -> avg=10 ; channel 1: all 20s -> avg=20
src = np.zeros((H, W, C), dtype=np.int8)
src[:, :, 0] = 10
src[:, :, 1] = 20
in_buf.write_array(src)
in_buf.sync_to_device()

sentinel = np.full(C, 0x55, dtype=np.int8)
out_buf.write_array(sentinel)
out_buf.sync_to_device()

before = out_buf.read_array((C,), np.int8)
print(f"pool_engine (smartconnect_hp1) test")
print(f"Output buffer BEFORE run: {before.tolist()}")

ip = PoolEngine()
elapsed = ip.run(in_buf.phys_addr, out_buf.phys_addr, H, W, C,
                  fp_in=0, fp_out=0)
ip.close()
print(f"HW call completed in {elapsed*1000:.4f} ms")

out_buf.sync_from_device()
after = out_buf.read_array((C,), np.int8)
print(f"Output buffer AFTER  run: {after.tolist()}")
print(f"Expected (approx):        [10, 20]")

if np.array_equal(after, sentinel):
    print("\n=> UNCHANGED -- same failure as head_transpose.")
    print("   Bug is SYSTEMIC across the whole design, not isolated")
    print("   to one smartconnect.")
else:
    print(f"\n=> CHANGED to {after.tolist()} -- pool_engine's write DID land!")
    print("   This means the bug is isolated to smartconnect_hp2 or")
    print("   head_transpose specifically, not the whole design.")

in_buf.close()
out_buf.close()
