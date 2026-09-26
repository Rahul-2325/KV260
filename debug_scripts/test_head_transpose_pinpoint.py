#!/usr/bin/env python3
"""
test_head_transpose_pinpoint.py
Same test as test_head_transpose_debug.py, but instrumented with a
print+flush after EVERY individual step -- including each register write
normally hidden inside HeadTranspose.run() -- to pinpoint exactly which
call hangs/crashes the board. Written after 3 crashes in one session
where the plain version only ever showed phys_addr printed, then nothing.

Does NOT use ip_wrappers.HeadTranspose -- talks to the two AXI-Lite
regions directly so each register write can be isolated and flushed
individually.
"""
import sys
from hw_ip_driver import ZoclBuffer, DRM_PATH_DEFAULT, AxiLiteRegion, poll_ap_done, IP_ADDR, AP_START
import numpy as np

def step(msg):
    print(msg, flush=True)

H, W, C = 4, 4, 3
total = H * W * C
src = np.arange(total, dtype=np.int8)

step("[1] allocating buffers...")
in_buf  = ZoclBuffer(DRM_PATH_DEFAULT, total)
out_buf = ZoclBuffer(DRM_PATH_DEFAULT, total)
step(f"[1] OK - in_buf.phys_addr={hex(in_buf.phys_addr)} out_buf.phys_addr={hex(out_buf.phys_addr)}")

step("[2] write_array...")
in_buf.write_array(src)
step("[2] OK")

step("[3] sync_to_device...")
in_buf.sync_to_device()
step("[3] OK")

step("[4] opening control AxiLiteRegion (mmap)...")
addr = IP_ADDR["head_transpose"]
ctrl = AxiLiteRegion(addr["control"])
step("[4] OK")

step("[5] opening control_r AxiLiteRegion (mmap)...")
ctrl_r = AxiLiteRegion(addr["control_r"])
step("[5] OK")

step("[6] write64_split data_in_addr into control_r[0x10]...")
ctrl_r.write64_split(0x10, in_buf.phys_addr)
step("[6] OK")

step("[7] write64_split data_out_addr into control_r[0x1c]...")
ctrl_r.write64_split(0x1c, out_buf.phys_addr)
step("[7] OK")

step("[8] write32 H into control[0x10]...")
ctrl.write32(0x10, H)
step("[8] OK")

step("[9] write32 W into control[0x18]...")
ctrl.write32(0x18, W)
step("[9] OK")

step("[10] write32 C into control[0x20]...")
ctrl.write32(0x20, C)
step("[10] OK")

step("[11] *** about to write AP_START (0x1) into control[0x00] -- this is the actual trigger ***")
ctrl.write32(0x00, AP_START)
step("[11] OK - AP_START write returned, IP should now be executing")

step("[12] polling ap_ctrl for AP_DONE (timeout 10s)...")
elapsed = poll_ap_done(ctrl, timeout_s=10.0)
step(f"[12] OK - AP_DONE seen after {elapsed*1000:.4f} ms")

step("[13] sync_from_device...")
out_buf.sync_from_device()
step("[13] OK")

step("[14] read_array...")
result = out_buf.read_array((H, W, C), np.int8)
step(f"[14] OK - HW result: {result.flatten().tolist()}")
print("Expected:  [0, 16, 32, 1, 17, 33, 2, 18, 34, 3, 19, 35, 4, 20, 36, 5, 21, 37, 6, 22, 38, 7, 23, 39, 8, 24, 40, 9, 25, 41, 10, 26, 42, 11, 27, 43, 12, 28, 44, 13, 29, 45, 14, 30, 46, 15, 31, 47]", flush=True)

ctrl.close()
ctrl_r.close()
in_buf.close()
out_buf.close()
step("[15] all closed cleanly - TEST COMPLETE")
