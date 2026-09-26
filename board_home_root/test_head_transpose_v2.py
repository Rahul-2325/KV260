#!/usr/bin/env python3
"""
test_head_transpose_v2.py
Corrected test for head_transpose AFTER the pragma fix (single unified
control interface, no more control_r split). Uses the REAL register
offsets from xhead_transpose_hw.h -- verified against the actual
synthesized IP, not guessed.
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
HPC_ALIAS_BIT = 0x8_0000_0000

class drm_zocl_create_bo(ctypes.Structure):
    _fields_ = [('size', ctypes.c_uint64), ('handle', ctypes.c_uint32), ('flags', ctypes.c_uint32)]
class drm_zocl_map_bo(ctypes.Structure):
    _fields_ = [('handle', ctypes.c_uint32), ('pad', ctypes.c_uint32), ('offset', ctypes.c_uint64)]
class drm_zocl_info_bo(ctypes.Structure):
    _fields_ = [('handle', ctypes.c_uint32), ('pad', ctypes.c_uint32), ('size', ctypes.c_uint64), ('paddr', ctypes.c_uint64)]

class ZoclBuffer:
    def __init__(self, drm_path, size):
        self.size = max(size, 4096)
        self._fd = os.open(drm_path, os.O_RDWR)
        create = drm_zocl_create_bo(size=self.size, handle=0, flags=ZOCL_BO_FLAGS_CMA)
        fcntl.ioctl(self._fd, ZOCL_IOCTL_CREATE_BO, create, True)
        self.handle = create.handle
        info = drm_zocl_info_bo(handle=self.handle, pad=0, size=0, paddr=0)
        fcntl.ioctl(self._fd, ZOCL_IOCTL_INFO_BO, info, True)
        raw_addr = info.paddr
        self.phys_addr = raw_addr - HPC_ALIAS_BIT if raw_addr >= HPC_ALIAS_BIT else raw_addr
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


# ---- REAL register map from xhead_transpose_hw.h (post pragma-fix) ----
CTRL_BASE = 0x80050000
R_AP_CTRL  = 0x00
R_DATA_IN  = 0x10   # 64-bit
R_DATA_OUT = 0x1c   # 64-bit
R_H        = 0x28
R_W        = 0x30
R_C        = 0x38
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


H, W, C = 4, 4, 3
total = H * W * C
src = np.arange(total, dtype=np.int8)

in_buf  = ZoclBuffer('/dev/dri/renderD128', total)
out_buf = ZoclBuffer('/dev/dri/renderD128', total)

print(f"in_buf.phys_addr  = {hex(in_buf.phys_addr)}")
print(f"out_buf.phys_addr = {hex(out_buf.phys_addr)}")

in_buf.write_array(src)

write64(CTRL_BASE, R_DATA_IN, in_buf.phys_addr)
write64(CTRL_BASE, R_DATA_OUT, out_buf.phys_addr)
write32(CTRL_BASE, R_H, H)
write32(CTRL_BASE, R_W, W)
write32(CTRL_BASE, R_C, C)
write32(CTRL_BASE, R_AP_CTRL, AP_START)

import time
start = time.perf_counter()
while True:
    val = read32(CTRL_BASE, R_AP_CTRL)
    if val & 0x2:
        break
    if time.perf_counter() - start > 5:
        print("TIMEOUT waiting for ap_done")
        break
elapsed = time.perf_counter() - start
print(f"HW call completed in {elapsed*1000:.4f} ms")

result = out_buf.read_array((H, W, C), np.int8)
print(f"HW result: {result.flatten().tolist()}")
print(f"Expected:  [0, 16, 32, 1, 17, 33, 2, 18, 34, 3, 19, 35, 4, 20, 36, 5, 21, 37, 6, 22, 38, 7, 23, 39, 8, 24, 40, 9, 25, 41, 10, 26, 42, 11, 27, 43, 12, 28, 44, 13, 29, 45, 14, 30, 46, 15, 31, 47]")

in_buf.close()
out_buf.close()
