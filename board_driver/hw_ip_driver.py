#!/usr/bin/env python3
"""
hw_ip_driver.py
Direct /dev/mem + zocl CMA driver for the 9 custom HLS IPs on the
kv260_custom_noDPU bitstream (no DPU, no VART, no XRT).

VERIFIED addresses (confirmed live on board — all 9 IPs read ap_ctrl=0x4
IDLE at these addresses):

IP                      control (scalars)   control_r (pointers)
conv2d_engine_0         0x8000_0000          0x8009_0000
depthwise_engine_0      0x8001_0000          0x800A_0000
eltwise_add_0           0x8002_0000          0x800B_0000
head_concat_reshape_0   0x8003_0000          0x800C_0000
head_sigmoid_0          0x8004_0000          0x800D_0000
head_transpose_0        0x8005_0000          0x800E_0000
lcam_attention_gate_0   0x8006_0000          0x800F_0000
pool_engine_0           0x8007_0000          0x8010_0000
upsample_engine_0       0x8008_0000          0x8011_0000

Each IP has TWO separate AXI-Lite interfaces (Vitis HLS auto-split
because our .cpp files bundled scalar args under "control" but did not
explicitly pragma the pointer args into the same bundle):
    control    -> ap_ctrl + all scalar (int) arguments
    control_r  -> the pointer (data_t*) arguments only, as 64-bit
                  lo/hi register pairs

Confirmed register offsets from Vitis-HLS-generated xconv2d_engine_hw.h
(same pattern applies to all 9 IPs -- control offsets start at 0x10 for
first scalar in declaration order, +8 each; control_r offsets start at
0x10 for first pointer, +12 each [4 lo + 4 hi + 4 reserved]).
"""

import mmap
import os
import struct
import time
import fcntl
import ctypes
import numpy as np

# ── Verified base addresses ─────────────────────────────────────────────────
# NOTE on "head_transpose": this control/control_r pair is the ORIGINAL
# two-interface register layout. head_transpose.cpp in hls_ip_sources/ was
# later modified to a single-interface layout as a debugging experiment (see
# PROJECT_HISTORY.md §23) -- that fix was tested and RULED OUT as the root
# cause, so it does not need to be propagated here unless you're specifically
# re-testing that exact experiment, in which case use
# debug_scripts/test_head_transpose_v2.py's register scheme instead (single
# CTRL_BASE, no control_r).
IP_ADDR = {
    "conv2d_engine":       {"control": 0x80000000, "control_r": 0x80090000},
    "depthwise_engine":    {"control": 0x80010000, "control_r": 0x800A0000},
    "eltwise_add":         {"control": 0x80020000, "control_r": 0x800B0000},
    "head_concat_reshape": {"control": 0x80030000, "control_r": 0x800C0000},
    "head_sigmoid":        {"control": 0x80040000, "control_r": 0x800D0000},
    "head_transpose":      {"control": 0x80050000, "control_r": 0x800E0000},
    "lcam_attention_gate": {"control": 0x80060000, "control_r": 0x800F0000},
    "pool_engine":         {"control": 0x80070000, "control_r": 0x80100000},
    "upsample_engine":     {"control": 0x80080000, "control_r": 0x80110000},
}
REG_WINDOW = 0x1_0000  # 64K per interface

# ── ap_ctrl bits (on the "control" interface only) ─────────────────────────
AP_START = 1 << 0
AP_DONE  = 1 << 1
AP_IDLE  = 1 << 2
AP_READY = 1 << 3
CTRL_OFFSET = 0x00


class AxiLiteRegion:
    """A single mmap'd 64K AXI-Lite register window over /dev/mem."""

    def __init__(self, base_addr: int, size: int = REG_WINDOW):
        self._fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self._mm = mmap.mmap(
            self._fd, size, mmap.MAP_SHARED,
            mmap.PROT_READ | mmap.PROT_WRITE, offset=base_addr
        )
        self.base_addr = base_addr

    def read32(self, offset: int) -> int:
        return struct.unpack_from("<I", self._mm, offset)[0]

    def write32(self, offset: int, value: int) -> None:
        struct.pack_into("<I", self._mm, offset, value & 0xFFFFFFFF)

    def write64_split(self, offset: int, addr64: int) -> None:
        """64-bit pointer: lo word at offset, hi word at offset+4."""
        self.write32(offset,     addr64 & 0xFFFFFFFF)
        self.write32(offset + 4, (addr64 >> 32) & 0xFFFFFFFF)

    def close(self):
        self._mm.close()
        os.close(self._fd)


def poll_ap_done(ctrl_region: AxiLiteRegion, timeout_s: float = 10.0,
                  poll_interval_s: float = 0.0002) -> float:
    """Poll ap_ctrl (on the CONTROL interface) until AP_DONE is set."""
    start = time.perf_counter()
    while True:
        val = ctrl_region.read32(CTRL_OFFSET)
        if val & AP_DONE:
            return time.perf_counter() - start
        if time.perf_counter() - start > timeout_s:
            raise TimeoutError(
                f"AP_DONE never asserted within {timeout_s}s "
                f"(last ap_ctrl read: {hex(val)}, "
                f"base={hex(ctrl_region.base_addr)})"
            )
        time.sleep(poll_interval_s)


# ============================================================================
# zocl DRM CMA buffer allocator
# ============================================================================
DRM_IOCTL_BASE = ord('d')
DRM_COMMAND_BASE = 0x40


def _IOWR(nr, size):
    return (3 << 30) | (size << 16) | (DRM_IOCTL_BASE << 8) | nr


ZOCL_IOCTL_CREATE_BO = _IOWR(DRM_COMMAND_BASE + 0, 16)
ZOCL_IOCTL_MAP_BO    = _IOWR(DRM_COMMAND_BASE + 3, 16)
ZOCL_IOCTL_SYNC_BO   = _IOWR(DRM_COMMAND_BASE + 4, 24)
ZOCL_IOCTL_INFO_BO   = _IOWR(DRM_COMMAND_BASE + 5, 24)
ZOCL_BO_FLAGS_CMA = 0x1 << 28

DRM_ZOCL_SYNC_BO_TO_DEVICE   = 0
DRM_ZOCL_SYNC_BO_FROM_DEVICE = 1


class drm_zocl_sync_bo(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32),
                ("dir", ctypes.c_uint32),
                ("offset", ctypes.c_uint64),
                ("size", ctypes.c_uint64)]


class drm_zocl_create_bo(ctypes.Structure):
    _fields_ = [("size", ctypes.c_uint64),
                ("handle", ctypes.c_uint32),
                ("flags", ctypes.c_uint32)]


class drm_zocl_map_bo(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32),
                ("pad", ctypes.c_uint32),
                ("offset", ctypes.c_uint64)]


class drm_zocl_info_bo(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32),
                ("pad", ctypes.c_uint32),
                ("size", ctypes.c_uint64),
                ("paddr", ctypes.c_uint64)]


# ── PL-reachable memory constraints (see PROJECT_HISTORY.md §26) ───────────
# This board has 4GB DDR split into TWO PHYSICALLY DISTINCT regions:
#     0x0000_0000 - 0x7FEF_FFFF   low  2GB
#     0x8_0000_0000 - 0x8_7FFF_FFFF  high 2GB
# They are NOT aliases of each other (proven by direct /dev/mem measurement).
# Our 9 IPs reach DRAM through S_AXI_HP0/1/2_FPD, whose address segments
# only decode 0x0-0x7FFF_FFFF -- so the PL can ONLY reach the LOW region.
#
# zocl ignores our MEM_TOPOLOGY ("[drm] Allocating BO from CMA for invalid
# or unused memory index[0]") and falls back to a generic coherent alloc.
# Empirically:
#     4KB  request -> generic page allocator -> HIGH DDR -> PL CANNOT REACH
#     64KB+ request -> the CMA pool (reserved low at 0x1600_0000) -> LOW  OK
# So we force every allocation to at least the CMA threshold, which lands
# all buffers in PL-reachable low DDR. This is what finally made the
# hardware produce correct output.
PL_REACHABLE_LIMIT = 0x8000_0000   # PL address decode ceiling
MIN_CMA_ALLOC      = 64 * 1024     # below this, allocs come from high DDR


class ZoclBuffer:
    """CMA-backed buffer via zocl DRM ioctls.

    NOTE: allocations are rounded up to MIN_CMA_ALLOC (64KB) so they are
    served from the low-DDR CMA pool the PL can actually reach. Requesting
    a smaller buffer silently yields a HIGH-DDR address that the PL cannot
    access, which manifests as all-zero / never-written output rather than
    any kind of error. See PROJECT_HISTORY.md §26.
    """

    def __init__(self, drm_path: str, size: int):
        self.size = max(size, MIN_CMA_ALLOC)
        self._fd = os.open(drm_path, os.O_RDWR)
        create = drm_zocl_create_bo(size=self.size, handle=0,
                                     flags=ZOCL_BO_FLAGS_CMA)
        fcntl.ioctl(self._fd, ZOCL_IOCTL_CREATE_BO, create, True)
        self.handle = create.handle

        info = drm_zocl_info_bo(handle=self.handle, pad=0, size=0, paddr=0)
        fcntl.ioctl(self._fd, ZOCL_IOCTL_INFO_BO, info, True)
        self.phys_addr = info.paddr

        mapreq = drm_zocl_map_bo(handle=self.handle, pad=0, offset=0)
        fcntl.ioctl(self._fd, ZOCL_IOCTL_MAP_BO, mapreq, True)
        self._mm = mmap.mmap(
            self._fd, self.size, mmap.MAP_SHARED,
            mmap.PROT_READ | mmap.PROT_WRITE, offset=mapreq.offset
        )

        # ── PL reachability check ───────────────────────────────────────
        # HISTORICAL BUG (Day 5 - 2026-08-29, PROJECT_HISTORY.md §26):
        # this used to SUBTRACT 0x8_0000_0000 here, on the assumption that
        # a high address was a coherent-port "alias" of the same low DRAM.
        # That assumption is FALSE -- high and low DDR are two physically
        # distinct regions. Subtracting the bit pointed the PL at real but
        # completely unrelated memory, which is why writes appeared to
        # succeed on the AXI bus (§17), captured addresses "matched" (§18),
        # read data looked like random junk (§19), and outputs were always
        # zero: the CPU and the PL were never touching the same memory.
        #
        # With MIN_CMA_ALLOC in place, allocations land in low DDR and this
        # never trips. If it ever does, FAIL LOUDLY rather than silently
        # corrupting the address -- a wrong address here is invisible at
        # runtime and costs days to diagnose.
        if self.phys_addr >= PL_REACHABLE_LIMIT:
            self.close()
            raise MemoryError(
                f"zocl returned phys_addr {hex(self.phys_addr)} in HIGH DDR, "
                f"which the PL's HP ports cannot reach (limit "
                f"{hex(PL_REACHABLE_LIMIT)}). Requested size {size} -> "
                f"allocated {self.size}. Do NOT strip the high bit: it is a "
                f"distinct memory region, not an alias. See PROJECT_HISTORY.md §26."
            )

    def write(self, data: bytes, offset: int = 0):
        self._mm[offset:offset + len(data)] = data

    def write_array(self, arr: np.ndarray, offset: int = 0):
        self.write(np.ascontiguousarray(arr).tobytes(), offset)

    def sync_to_device(self, offset: int = 0, size: int = None):
        """Flush CPU cache -> DRAM. Call AFTER a CPU write, BEFORE
        triggering a PL master to read this buffer."""
        sz = self.size if size is None else size
        s = drm_zocl_sync_bo(handle=self.handle,
                              dir=DRM_ZOCL_SYNC_BO_TO_DEVICE,
                              offset=offset, size=sz)
        fcntl.ioctl(self._fd, ZOCL_IOCTL_SYNC_BO, s, True)

    def sync_from_device(self, offset: int = 0, size: int = None):
        """Invalidate CPU cache, forcing a fresh read from DRAM. Call
        AFTER a PL master has written this buffer, BEFORE reading it
        back from the CPU side."""
        sz = self.size if size is None else size
        s = drm_zocl_sync_bo(handle=self.handle,
                              dir=DRM_ZOCL_SYNC_BO_FROM_DEVICE,
                              offset=offset, size=sz)
        fcntl.ioctl(self._fd, ZOCL_IOCTL_SYNC_BO, s, True)

    def read(self, nbytes: int, offset: int = 0) -> bytes:
        return bytes(self._mm[offset:offset + nbytes])

    def read_array(self, shape, dtype, offset: int = 0) -> np.ndarray:
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        raw = self.read(nbytes, offset)
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()

    def close(self):
        self._mm.close()
        os.close(self._fd)


DRM_PATH_DEFAULT = "/dev/dri/renderD128"
