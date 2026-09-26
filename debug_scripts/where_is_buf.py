import mmap, os, struct, fcntl, ctypes
import numpy as np
DRM_IOCTL_BASE = ord('d'); DRM_COMMAND_BASE = 0x40
def _IOWR(nr, size): return (3 << 30) | (size << 16) | (DRM_IOCTL_BASE << 8) | nr
CREATE=_IOWR(DRM_COMMAND_BASE+0,16); MAPBO=_IOWR(DRM_COMMAND_BASE+3,16); INFO=_IOWR(DRM_COMMAND_BASE+5,24)
CMA = 0x1 << 28
class C_(ctypes.Structure): _fields_=[('size',ctypes.c_uint64),('handle',ctypes.c_uint32),('flags',ctypes.c_uint32)]
class M_(ctypes.Structure): _fields_=[('handle',ctypes.c_uint32),('pad',ctypes.c_uint32),('offset',ctypes.c_uint64)]
class I_(ctypes.Structure): _fields_=[('handle',ctypes.c_uint32),('pad',ctypes.c_uint32),('size',ctypes.c_uint64),('paddr',ctypes.c_uint64)]

fd = os.open('/dev/dri/renderD128', os.O_RDWR)
c = C_(size=4096, handle=0, flags=CMA); fcntl.ioctl(fd, CREATE, c, True)
i = I_(handle=c.handle, pad=0, size=0, paddr=0); fcntl.ioctl(fd, INFO, i, True)
raw = i.paddr
m = M_(handle=c.handle, pad=0, offset=0); fcntl.ioctl(fd, MAPBO, m, True)
mm = mmap.mmap(fd, 4096, mmap.MAP_SHARED, mmap.PROT_READ|mmap.PROT_WRITE, offset=m.offset)

low = raw - 0x800000000 if raw >= 0x800000000 else raw
print("zocl reported paddr = 0x%x" % raw)
print("stripped (low)      = 0x%x" % low)

PAT = b'\xAB\xCD\xEF\x99'
mm[0:4] = PAT
print("wrote pattern ABCDEF99 via zocl mmap; readback via zocl = %s" % mm[0:4].hex())

def peek(phys):
    try:
        f = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
        g = mmap.mmap(f, 4096, mmap.MAP_SHARED, mmap.PROT_READ, offset=phys)
        v = g[0:4].hex(); g.close(); os.close(f); return v
    except Exception as e:
        return "ERROR: %s" % e

print("/dev/mem @ HIGH 0x%x = %s" % (raw, peek(raw)))
print("/dev/mem @ LOW  0x%x = %s" % (low, peek(low)))
mm.close(); os.close(fd)
