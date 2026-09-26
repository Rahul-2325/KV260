#!/usr/bin/env python3
"""
dump_dpu_regs.py
Read-only dump of the DPU's AXI-Lite register space.

Diagnosing: VART reports `dpu timeout! core_idx = 0` with every profiling
counter reading zero (LSTART/LEND/CSTART/... all 0), and xdputil reports
IP version v0.0.0 / timestamp 200-00-00 / frequency 0. Those registers are
hardwired constants baked in at synthesis, so:

  * if they read as REAL values  -> the DPU core is clocked and alive;
    the problem is in how the CU is being started / completion signalled.
  * if they read as ZERO         -> the DPU's core clock domain is dead
    (dpu_clk_wiz not locked), even though the AXI-Lite side responds on
    s_axi_aclk = pl_clk0.

Purely reads -- no writes, no DMA, cannot hang the board.
"""
import mmap, os, struct

BASE = 0x8F000000
SPAN = 0x1000


def dump(base, span):
    fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
    mm = mmap.mmap(fd, span, mmap.MAP_SHARED, mmap.PROT_READ, offset=base)
    vals = {}
    for off in range(0, span, 4):
        vals[off] = struct.unpack_from('<I', mm, off)[0]
    mm.close()
    os.close(fd)
    return vals


v = dump(BASE, SPAN)
nz = {o: x for o, x in v.items() if x not in (0x00000000, 0xFFFFFFFF)}
print("DPU @ 0x%08X : %d/%d registers are non-zero/non-FF"
      % (BASE, len(nz), len(v)))
print()
print("non-zero registers:")
for o in sorted(nz):
    print("  0x%03X = 0x%08X" % (o, nz[o]))
print()
print("first 32 words:")
for o in range(0, 128, 4):
    print("  0x%03X = 0x%08X" % (o, v[o]))
