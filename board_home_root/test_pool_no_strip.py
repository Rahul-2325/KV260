#!/usr/bin/env python3
"""
test_pool_no_strip.py
Tests pool_engine (still on the ORIGINAL non-coherent smartconnect_hp1 path,
never touched by yesterday's HPC0 rewiring) using the FULL, UN-STRIPPED
physical address instead of subtracting the 0x800000000 "alias" bit.

This directly tests the Day-6 hypothesis: the HPC_ALIAS_BIT subtraction
introduced on Day 5 may have been WRONG -- /proc/iomem shows two
SEPARATE System RAM regions (low 2GB and a high 2GB+ starting at 32GB),
not one aliased region. If that's right, subtracting the bit sends the
hardware to a different, unrelated (but still valid) piece of memory.
"""
import mmap, os, struct, fcntl, ctypes
import numpy as np

DRM_IOCTL_BASE = ord('d')
DRM_COMMAND_BASE = 0x40
def _IOWR(nr, size):
    return (3 << 30) | (size << 16) | (DRM_IOCTL_BASE << 8) | nr

ZOCL_IOCTL_CREATE_BO = _IOWR(DRM_COMMAND_BASE + 0, 16)
ZOCL_IOCTL_MAP_BO    = _IOWR(DRM_COMMAND_BASE + 3, 16)
ZOCL_IOCTL_INFO_BO   = _IOWR(DRM_COMMAND_BASE + 5, 24)
ZOCL_BO_FLAGS_CMA = 0x1 << 28

class drm_zocl_create_bo(ctypes.Structure):
    _fields_ = [('size', ctypes.c_uint64), ('handle', ctypes.c_uint32), ('flags', ctypes.c_uint32)]
class drm_zocl_map_bo(ctypes.Structure):
    _fields_ = [('handle', ctypes.c_uint32), ('pad', ctypes.c_uint32), ('offset', ctypes.c_uint64)]
class drm_zocl_info_bo(ctypes.Structure):
    _fields_ = [('handle', ctypes.c_uint32), ('pad', ctypes.c_uint32), ('size', ctypes.c_uint64), ('paddr', ctypes.c_uint64)]

class NoStripBuffer:
    """Same as ZoclBuffer but WITHOUT stripping the high-address bit."""
    def __init__(self, drm_path, size):
        self.size = max(size, 4096)
        self._fd = os.open(drm_path, os.O_RDWR)
        create = drm_zocl_create_bo(size=self.size, handle=0, flags=ZOCL_BO_FLAGS_CMA)
        fcntl.ioctl(self._fd, ZOCL_IOCTL_CREATE_BO, create, True)
        self.handle = create.handle

        info = drm_zocl_info_bo(handle=self.handle, pad=0, size=0, paddr=0)
        fcntl.ioctl(self._fd, ZOCL_IOCTL_INFO_BO, info, True)
        self.phys_addr = info.paddr   # <-- NOT stripping anything this time

        mapreq = drm_zocl_map_bo(handle=self.handle, pad=0, offset=0)
        fcntl.ioctl(self._fd, ZOCL_IOCTL_MAP_BO, mapreq, True)
        self._mm = mmap.mmap(self._fd, self.size, mmap.MAP_SHARED,
                              mmap.PROT_READ | mmap.PROT_WRITE, offset=mapreq.offset)

    def write_array(self, arr):
        self._mm[0:arr.nbytes] = np.ascontiguousarray(arr).tobytes()

    def read_array(self, shape, dtype):
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        return np.frombuffer(self._mm[0:nbytes], dtype=dtype).reshape(shape).copy()

    def close(self):
        self._mm.close()
        os.close(self._fd)


# ---- pool_engine control register offsets (verified real header) ----
CTRL_BASE   = 0x80070000
CTRL_R_BASE = 0x80100000
R_H = 0x10; R_W = 0x18; R_C = 0x20; R_FP_IN = 0x28; R_FP_OUT = 0x30
R_FEAT_IN = 0x10; R_FEAT_OUT = 0x1c
AP_START = 1

def write32(base, off, val):
    fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
    mm = mmap.mmap(fd, 0x1000, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=base)
    struct.pack_into('<I', mm, off, val & 0xFFFFFFFF)
    mm.close(); os.close(fd)

def read32(base, off):
    fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
    mm = mmap.mmap(fd, 0x1000, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=base)
    val = struct.unpack_from('<I', mm, off)[0]
    mm.close(); os.close(fd)
    return val

def write64(base, off, val):
    write32(base, off, val & 0xFFFFFFFF)
    write32(base, off + 4, (val >> 32) & 0xFFFFFFFF)


H, W, C = 4, 4, 2
n_in = H * W * C

in_buf  = NoStripBuffer('/dev/dri/renderD128', n_in)
out_buf = NoStripBuffer('/dev/dri/renderD128', C)

print(f"in_buf.phys_addr  = {hex(in_buf.phys_addr)}  (FULL, not stripped)")
print(f"out_buf.phys_addr = {hex(out_buf.phys_addr)}  (FULL, not stripped)")

src = np.zeros((H, W, C), dtype=np.int8)
src[:, :, 0] = 10
src[:, :, 1] = 20
in_buf.write_array(src)

sentinel = np.full(C, 0x55, dtype=np.int8)
out_buf.write_array(sentinel)

write64(CTRL_R_BASE, R_FEAT_IN, in_buf.phys_addr)
write64(CTRL_R_BASE, R_FEAT_OUT, out_buf.phys_addr)
write32(CTRL_BASE, R_H, H)
write32(CTRL_BASE, R_W, W)
write32(CTRL_BASE, R_C, C)
write32(CTRL_BASE, R_FP_IN, 0)
write32(CTRL_BASE, R_FP_OUT, 0)
write32(CTRL_BASE, 0x00, AP_START)

import time
start = time.perf_counter()
while True:
    val = read32(CTRL_BASE, 0x00)
    if val & 0x2:
        break
    if time.perf_counter() - start > 5:
        print("TIMEOUT waiting for ap_done")
        break
elapsed = time.perf_counter() - start
print(f"HW call completed in {elapsed*1000:.4f} ms")

result = out_buf.read_array((C,), np.int8)
print(f"HW result: {result.tolist()}")
print(f"Expected:  [10, 20]  (approx -- average pooling)")

in_buf.close()
out_buf.close()
