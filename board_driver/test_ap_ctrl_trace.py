#!/usr/bin/env python3
"""
test_ap_ctrl_trace.py
Traces the ap_ctrl register bit-by-bit through a full run, instead of
just waiting for ap_done. This shows us whether the IP genuinely
starts executing (ap_idle drops to 0) or whether ap_start is being
ignored/misrouted entirely.

ap_ctrl bits: bit0=ap_start bit1=ap_done bit2=ap_idle bit3=ap_ready
"""
import time
from hw_ip_driver import AxiLiteRegion, ZoclBuffer, DRM_PATH_DEFAULT, IP_ADDR

def decode(v):
    bits = []
    if v & 0x1: bits.append("START")
    if v & 0x2: bits.append("DONE")
    if v & 0x4: bits.append("IDLE")
    if v & 0x8: bits.append("READY")
    return f"{hex(v)} [{','.join(bits) if bits else 'none'}]"

addr = IP_ADDR["head_transpose"]
ctrl   = AxiLiteRegion(addr["control"])
ctrl_r = AxiLiteRegion(addr["control_r"])

H, W, C = 4, 4, 3
total = H * W * C

in_buf  = ZoclBuffer(DRM_PATH_DEFAULT, total)
out_buf = ZoclBuffer(DRM_PATH_DEFAULT, total)
in_buf.write(bytes(range(total)))

print(f"BEFORE  ap_ctrl = {decode(ctrl.read32(0x00))}")

# Write pointers first
ctrl_r.write32(0x10, in_buf.phys_addr & 0xFFFFFFFF)
ctrl_r.write32(0x14, (in_buf.phys_addr >> 32) & 0xFFFFFFFF)
ctrl_r.write32(0x1c, out_buf.phys_addr & 0xFFFFFFFF)
ctrl_r.write32(0x20, (out_buf.phys_addr >> 32) & 0xFFFFFFFF)
print(f"in_buf.phys_addr  = {hex(in_buf.phys_addr)}")
print(f"out_buf.phys_addr = {hex(out_buf.phys_addr)}")
print(f"control_r readback: DATA_IN_lo={hex(ctrl_r.read32(0x10))} "
      f"DATA_IN_hi={hex(ctrl_r.read32(0x14))}")
print(f"control_r readback: DATA_OUT_lo={hex(ctrl_r.read32(0x1c))} "
      f"DATA_OUT_hi={hex(ctrl_r.read32(0x20))}")

# Write scalars
ctrl.write32(0x10, H)
ctrl.write32(0x18, W)
ctrl.write32(0x20, C)
print(f"control readback: H={ctrl.read32(0x10)} W={ctrl.read32(0x18)} "
      f"C={ctrl.read32(0x20)}")

# Trigger
ctrl.write32(0x00, 0x1)  # ap_start
print(f"AFTER START ap_ctrl = {decode(ctrl.read32(0x00))}")

for i in range(20):
    v = ctrl.read32(0x00)
    print(f"  t+{i}: ap_ctrl = {decode(v)}")
    if v & 0x2:
        print("  -> ap_done seen, stopping trace")
        break
    time.sleep(0.001)

result = out_buf.read_array((H, W, C), "int8")
print(f"\nOutput buffer content: {result.flatten().tolist()}")

ctrl.close()
ctrl_r.close()
in_buf.close()
out_buf.close()
