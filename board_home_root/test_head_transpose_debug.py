#!/usr/bin/env python3
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

ip = HeadTranspose()
elapsed = ip.run(in_buf.phys_addr, out_buf.phys_addr, H, W, C)
ip.close()
print(f"HW call completed in {elapsed*1000:.4f} ms")

out_buf.sync_from_device()
result = out_buf.read_array((H, W, C), np.int8)
print(f"HW result: {result.flatten().tolist()}")

in_buf.close()
out_buf.close()
