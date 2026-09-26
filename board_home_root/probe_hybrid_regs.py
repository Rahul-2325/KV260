#!/usr/bin/env python3
"""
probe_hybrid_regs.py
SAFE read-only register probe of the hybrid bitstream (DPU + LCAM IP).

Run this BEFORE any DMA or DPU workload. A plain AXI-Lite read is the
cheapest way to confirm both blocks are alive at the addresses the device
tree claims. If an address is wrong the read returns garbage (or hangs),
which is far better to discover here than mid-inference -- an unanswered
AXI access hangs the interconnect and reboots the board (§25/§29.7).

Expected:
  lcam s_axi_control @0x8006_0000 -> ap_ctrl == 0x4 (AP_IDLE)
  DPU                @0x8F00_0000 -> some non-0xFFFFFFFF value
"""
import mmap, os, struct


def rd(base, off=0):
    fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
    mm = mmap.mmap(fd, 0x1000, mmap.MAP_SHARED, mmap.PROT_READ, offset=base)
    v = struct.unpack_from('<I', mm, off)[0]
    mm.close()
    os.close(fd)
    return v


print("lcam s_axi_control   @0x80060000 ap_ctrl = 0x%08x   (expect 0x4 = IDLE)"
      % rd(0x80060000))
print("lcam s_axi_control_r @0x800F0000 [0x00]  = 0x%08x" % rd(0x800F0000))
print("DPU                  @0x8F000000 [0x00]  = 0x%08x" % rd(0x8F000000))
print("DPU                  @0x8F000000 [0x1C]  = 0x%08x" % rd(0x8F000000, 0x1C))
