#!/usr/bin/env python3
"""
test_control_r_readback.py
Writes a known pattern directly into head_transpose's control_r
registers (the pointer args) and reads it straight back -- WITHOUT
ever triggering ap_start. This isolates whether the control_r
AXI-Lite interface is even reachable at all, independent of the IP's
actual compute logic or the AXI master (data) path.
"""
from hw_ip_driver import AxiLiteRegion, IP_ADDR

addr = IP_ADDR["head_transpose"]
ctrl_r = AxiLiteRegion(addr["control_r"])

test_val_lo = 0xDEADBEEF
test_val_hi = 0x00000007

print(f"control_r base = {hex(addr['control_r'])}")

ctrl_r.write32(0x10, test_val_lo)   # DATA_IN lo
ctrl_r.write32(0x14, test_val_hi)   # DATA_IN hi

readback_lo = ctrl_r.read32(0x10)
readback_hi = ctrl_r.read32(0x14)

print(f"Wrote : lo={hex(test_val_lo)} hi={hex(test_val_hi)}")
print(f"Readback: lo={hex(readback_lo)} hi={hex(readback_hi)}")

if readback_lo == test_val_lo and readback_hi == test_val_hi:
    print("PASS -- control_r registers are reachable and read/write correctly")
else:
    print("FAIL -- control_r interface is not responding as expected")
    print("This points to a block-design connectivity issue on the")
    print("control_r AXI-Lite path (smartconnect_ctrl M09-M17), not")
    print("the data/buffer path we already fixed.")

ctrl_r.close()
