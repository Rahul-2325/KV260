import mmap, os, fcntl, ctypes
DRM_IOCTL_BASE = ord('d'); DRM_COMMAND_BASE = 0x40
def _IOWR(nr, size): return (3 << 30) | (size << 16) | (DRM_IOCTL_BASE << 8) | nr
CREATE=_IOWR(DRM_COMMAND_BASE+0,16); INFO=_IOWR(DRM_COMMAND_BASE+5,24)
CMA = 0x1 << 28
class C_(ctypes.Structure): _fields_=[('size',ctypes.c_uint64),('handle',ctypes.c_uint32),('flags',ctypes.c_uint32)]
class I_(ctypes.Structure): _fields_=[('handle',ctypes.c_uint32),('pad',ctypes.c_uint32),('size',ctypes.c_uint64),('paddr',ctypes.c_uint64)]
LIMIT = 0x80000000
print("%-12s %-16s %s" % ("SIZE", "PADDR", "REACHABLE BY PL (<0x80000000)?"))
for sz in [4096, 64*1024, 1024*1024, 4*1024*1024, 16*1024*1024, 64*1024*1024, 256*1024*1024]:
    try:
        fd = os.open('/dev/dri/renderD128', os.O_RDWR)
        c = C_(size=sz, handle=0, flags=CMA); fcntl.ioctl(fd, CREATE, c, True)
        i = I_(handle=c.handle, pad=0, size=0, paddr=0); fcntl.ioctl(fd, INFO, i, True)
        ok = "YES  <<<<" if i.paddr < LIMIT else "no (high DDR)"
        print("%-12s 0x%-14x %s" % (sz, i.paddr, ok))
        os.close(fd)
    except Exception as e:
        print("%-12s FAILED: %s" % (sz, e))
