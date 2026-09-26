#!/usr/bin/env python3
"""
test_sentinel.py
Pre-fills the OUTPUT buffer with a known non-zero sentinel (0x55)
BEFORE triggering the IP. This disambiguates two very different
failure modes that both otherwise look like "all zeros":

  - If output buffer STILL reads 0x55 after run -> the AXI WRITE
    (output) side never actually reached DRAM at all.
  - If output buffer reads 0x00 (or anything else) after run -> the
    IP DID write to DRAM, meaning the write path works, and the
    zeros must be coming from the IP reading zero on its INPUT side
    (read path problem instead).
"""
from hw_ip_driver import ZoclBuffer, DRM_PATH_DEFAULT
from ip_wrappers import HeadTranspose
import numpy as np

H, W, C = 4, 4, 3
total = H * W * C

in_buf  = ZoclBuffer(DRM_PATH_DEFAULT, total)
out_buf = ZoclBuffer(DRM_PATH_DEFAULT, total)

# Known distinct input pattern
src = np.arange(total, dtype=np.int8)
in_buf.write_array(src)

# Sentinel fill on OUTPUT before running -- 0x55 = 85, never a value
# our transpose would naturally produce from input 0..47
sentinel = np.full(total, 0x55, dtype=np.int8)
out_buf.write_array(sentinel)

before = out_buf.read_array((total,), np.int8)
print(f"Output buffer BEFORE run: {before.tolist()}")

ip = HeadTranspose()
elapsed = ip.run(in_buf.phys_addr, out_buf.phys_addr, H, W, C)
ip.close()
print(f"HW call completed in {elapsed*1000:.4f} ms")

after = out_buf.read_array((total,), np.int8)
print(f"Output buffer AFTER  run: {after.tolist()}")

if np.array_equal(after, sentinel):
    print("\n=> WRITE SIDE NEVER FIRED. Sentinel unchanged.")
    print("   The AXI WRITE master (output) never reached DRAM.")
elif np.all(after == 0):
    print("\n=> Sentinel was overwritten with zeros.")
    print("   WRITE path works. The IP is reading zero on its INPUT")
    print("   side -- READ path is the problem, not write.")
else:
    print("\n=> Sentinel changed to something else entirely:")
    print(f"   {after.tolist()}")
    print("   Both paths are active but producing wrong/garbage data.")

in_buf.close()
out_buf.close()
