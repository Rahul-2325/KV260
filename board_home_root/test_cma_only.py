#!/usr/bin/env python3
"""
test_cma_only.py
Isolates the zocl CMA buffer mechanism from any HLS IP call.
Just allocates a buffer, writes via CPU mmap, reads back via CPU mmap.
If THIS fails or shows a weird phys_addr, the bug is in buffer
allocation, not in our IP register offsets or HW logic.
"""
from hw_ip_driver import ZoclBuffer, DRM_PATH_DEFAULT
import numpy as np

buf = ZoclBuffer(DRM_PATH_DEFAULT, 48)
print(f"Allocated buffer: phys_addr={hex(buf.phys_addr)} size={buf.size}")

data = np.arange(48, dtype=np.int8)
buf.write_array(data)
readback = buf.read_array((48,), np.int8)

print(f"Written : {data.tolist()}")
print(f"Readback: {readback.tolist()}")
print("PASS" if np.array_equal(data, readback) else "FAIL")

buf.close()
