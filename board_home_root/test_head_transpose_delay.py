#!/usr/bin/env python3
import time
from hw_ip_driver import ZoclBuffer, DRM_PATH_DEFAULT
from ip_wrappers import HeadTranspose
import numpy as np

H, W, C = 4, 4, 3
total = H * W * C
src = np.arange(total, dtype=np.int8)

in_buf  = ZoclBuffer(DRM_PATH_DEFAULT, total)
out_buf = ZoclBuffer(DRM_PATH_DEFAULT, total)

print(f"in_buf.phys_addr  = {hex(in_buf.phys_addr)}")
print(f"out_buf.phys_addr = {hex(out_buf.phys_addr)}")

in_buf.write_array(src)
in_buf.sync_to_device()

# DIAGNOSTIC: deliberate delay to test if this is a timing race
print("Sleeping 100ms before triggering IP...")
time.sleep(0.1)

ip = HeadTranspose()
elapsed = ip.run(in_buf.phys_addr, out_buf.phys_addr, H, W, C)
ip.close()
print(f"HW call completed in {elapsed*1000:.4f} ms")

out_buf.sync_from_device()
result = out_buf.read_array((H, W, C), np.int8)
print(f"HW result: {result.flatten().tolist()}")
print(f"Expected:  [0, 16, 32, 1, 17, 33, 2, 18, 34, 3, 19, 35, 4, 20, 36, 5, 21, 37, 6, 22, 38, 7, 23, 39, 8, 24, 40, 9, 25, 41, 10, 26, 42, 11, 27, 43, 12, 28, 44, 13, 29, 45, 14, 30, 46, 15, 31, 47]")

in_buf.close()
out_buf.close()
