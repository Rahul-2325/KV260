#!/usr/bin/env python3
"""
kv260_lcam_driver.py

Direct /dev/mem register driver for decode_0 and nms_top_0 on the
KV260 LCAM-YOLOX custom bitstream (kv260_lcam_FINAL.bit.bin).

Register maps confirmed from Vivado-generated xdecode_hw.h / xnms_top_hw.h.
Both IPs use ap_ctrl_hs (polling) — interrupt pins are unconnected by design.

Run as root (needs /dev/mem access).
"""

import mmap
import os
import struct
import time

# ---------------------------------------------------------------------------
# Confirmed physical addresses (from pl.dtsi / assign_bd_address output)
# ---------------------------------------------------------------------------
DECODE_BASE       = 0x8000_0000   # decode_0 s_axi_control
NMS_CTRL_BASE     = 0x8001_0000   # nms_top_0 s_axi_CTRL   (AP_CTRL, thresholds, num_out)
NMS_CONTROL_BASE  = 0x8002_0000   # nms_top_0 s_axi_control (boxes_in / boxes_out pointers)
REG_WINDOW        = 0x1_0000      # 64K each, matches Vivado address assignment

# ---------------------------------------------------------------------------
# decode_0 register offsets (xdecode_hw.h)
# ---------------------------------------------------------------------------
DECODE_AP_CTRL       = 0x00
DECODE_GIE            = 0x04
DECODE_IER            = 0x08
DECODE_ISR            = 0x0c
DECODE_IN_DATA_LO      = 0x10
DECODE_IN_DATA_HI      = 0x14
DECODE_OUT_DATA_LO     = 0x1c
DECODE_OUT_DATA_HI     = 0x20

# ---------------------------------------------------------------------------
# nms_top_0 s_axi_CTRL register offsets (xnms_top_hw.h)
# ---------------------------------------------------------------------------
NMS_CTRL_AP_CTRL        = 0x00
NMS_CTRL_GIE             = 0x04
NMS_CTRL_IER             = 0x08
NMS_CTRL_ISR             = 0x0c
NMS_CTRL_NUM_OUT_DATA    = 0x10   # read: number of detections written
NMS_CTRL_NUM_OUT_VLD     = 0x14   # read: bit0 = num_out valid (clear on read)
NMS_CTRL_CONF_THRESH     = 0x20   # write: ap_fixed<20,12>, 0x20 offset
NMS_CTRL_NMS_THRESH      = 0x28   # write: ap_fixed<20,12>, 0x28 offset

# ---------------------------------------------------------------------------
# nms_top_0 s_axi_control register offsets (the s_axi_control fix)
# ---------------------------------------------------------------------------
NMS_CONTROL_BOXES_IN_LO   = 0x10
NMS_CONTROL_BOXES_IN_HI   = 0x14
NMS_CONTROL_BOXES_OUT_LO  = 0x1c
NMS_CONTROL_BOXES_OUT_HI  = 0x20

# AP_CTRL bit positions (shared by both interfaces, both IPs)
AP_START = 1 << 0
AP_DONE  = 1 << 1
AP_IDLE  = 1 << 2
AP_READY = 1 << 3

# ap_fixed<20,12> scale: 12 integer bits, 8 fractional bits (20-bit total)
# value_raw = round(value_float * 2**8)
FIXED_FRAC_BITS = 8


def to_fixed(value: float) -> int:
    """Convert a float to the ap_fixed<20,12> raw integer representation."""
    return int(round(value * (1 << FIXED_FRAC_BITS))) & 0xFFFFF


class AxiLiteRegion:
    """
    A single mmap'd 64K AXI-Lite register window over /dev/mem.
    Keep the mmap open for the lifetime of the driver -- opening/closing
    /dev/mem per access is unnecessary overhead and risks partial mappings.
    """

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

    def write64_split(self, offset_lo: int, offset_hi: int, addr64: int) -> None:
        """AXI pointer args are exposed as two 32-bit registers (lo/hi)."""
        self.write32(offset_lo, addr64 & 0xFFFFFFFF)
        self.write32(offset_hi, (addr64 >> 32) & 0xFFFFFFFF)

    def close(self):
        self._mm.close()
        os.close(self._fd)


def poll_ap_done(region: AxiLiteRegion, ap_ctrl_offset: int,
                  timeout_s: float = 2.0, poll_interval_s: float = 0.0002) -> float:
    """
    Poll AP_CTRL until AP_DONE is set. Returns elapsed wall time in seconds.
    Raises TimeoutError if the IP never completes -- this usually means a bad
    buffer pointer, a clock/reset issue, or the IP was never actually started.
    """
    start = time.perf_counter()
    while True:
        val = region.read32(ap_ctrl_offset)
        if val & AP_DONE:
            return time.perf_counter() - start
        if time.perf_counter() - start > timeout_s:
            raise TimeoutError(
                f"AP_DONE never asserted within {timeout_s}s "
                f"(last AP_CTRL read: {hex(val)})"
            )
        time.sleep(poll_interval_s)


import fcntl
import ctypes

# ---------------------------------------------------------------------------
# zocl DRM buffer-object allocator (bypasses XRT/pyxrt entirely).
#
# zocl exposes CMA-backed buffer allocation through the generic DRM ioctl
# interface (the same subsystem GPUs use). This gives us a physically
# contiguous, pinned buffer with a real device-side (bus) address that
# decode_0 / nms_top_0's AXI masters can actually dereference -- something
# plain Python/numpy heap memory cannot provide.
#
# Struct layouts and ioctl numbers taken from zocl's own driver headers
# (drm_zocl_create_bo / drm_zocl_info_bo / drm_zocl_map_bo), matching the
# ZOCL Driver Interfaces documentation for XRT 2022.1/2022.2.
# ---------------------------------------------------------------------------

DRM_IOCTL_BASE = ord('d')
DRM_COMMAND_BASE = 0x40

def _IOWR(nr, size):
    # Linux _IOWR macro: dir=3 (read+write), type='d', nr, size
    return (3 << 30) | (size << 16) | (DRM_IOCTL_BASE << 8) | nr

# zocl-specific ioctl numbers (offset from DRM_COMMAND_BASE); these match
# the drm_zocl_ioctl table -- CREATE_BO=0, MAP_BO=1, SYNC_BO=2, INFO_BO=3
ZOCL_IOCTL_CREATE_BO = _IOWR(DRM_COMMAND_BASE + 0, 24)   # struct drm_zocl_create_bo
ZOCL_IOCTL_MAP_BO    = _IOWR(DRM_COMMAND_BASE + 1, 16)   # struct drm_zocl_map_bo
ZOCL_IOCTL_INFO_BO   = _IOWR(DRM_COMMAND_BASE + 3, 24)   # struct drm_zocl_info_bo

ZOCL_BO_FLAGS_CMA = 0x00000001  # allocate from CMA pool


class drm_zocl_create_bo(ctypes.Structure):
    _fields_ = [
        ("size",  ctypes.c_uint64),
        ("handle", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
    ]


class drm_zocl_map_bo(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("pad",    ctypes.c_uint32),
        ("offset", ctypes.c_uint64),
    ]


class drm_zocl_info_bo(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("pad",    ctypes.c_uint32),
        ("size",    ctypes.c_uint64),
        ("paddr",   ctypes.c_uint64),
    ]


class ZoclBuffer:
    """
    A single CMA-backed buffer allocated through zocl's DRM ioctls.
    Provides both a CPU-writable mmap view and the physical (device) address
    to hand to decode_0 / nms_top_0's AXI pointer registers.
    """

    def __init__(self, drm_path: str, size: int):
        self.size = size
        self._fd = os.open(drm_path, os.O_RDWR)

        create = drm_zocl_create_bo(size=size, handle=0, flags=ZOCL_BO_FLAGS_CMA)
        fcntl.ioctl(self._fd, ZOCL_IOCTL_CREATE_BO, create, True)
        self.handle = create.handle

        info = drm_zocl_info_bo(handle=self.handle, pad=0, size=0, paddr=0)
        fcntl.ioctl(self._fd, ZOCL_IOCTL_INFO_BO, info, True)
        self.phys_addr = info.paddr

        mapreq = drm_zocl_map_bo(handle=self.handle, pad=0, offset=0)
        fcntl.ioctl(self._fd, ZOCL_IOCTL_MAP_BO, mapreq, True)

        self._mm = mmap.mmap(
            self._fd, size, mmap.MAP_SHARED,
            mmap.PROT_READ | mmap.PROT_WRITE, offset=mapreq.offset
        )

    def write(self, data: bytes, offset: int = 0):
        self._mm[offset:offset + len(data)] = data

    def read(self, nbytes: int, offset: int = 0) -> bytes:
        return bytes(self._mm[offset:offset + nbytes])

    def close(self):
        self._mm.close()
        os.close(self._fd)


class Decode:
    """Driver for decode_0 -- INT8 DPU-output -> fixed-point box decode."""

    def __init__(self):
        self.region = AxiLiteRegion(DECODE_BASE)

    def run(self, in_data_addr: int, out_data_addr: int) -> float:
        """
        Trigger one decode pass. in_data_addr / out_data_addr are physical
        DRAM addresses (not virtual pointers) -- these must be addresses the
        DPU/decode IP's own AXI masters can reach, i.e. real physical memory,
        typically allocated via a DMA-capable allocator (see note in main()).
        """
        self.region.write64_split(DECODE_IN_DATA_LO, DECODE_IN_DATA_HI, in_data_addr)
        self.region.write64_split(DECODE_OUT_DATA_LO, DECODE_OUT_DATA_HI, out_data_addr)
        self.region.write32(DECODE_AP_CTRL, AP_START)
        elapsed = poll_ap_done(self.region, DECODE_AP_CTRL)
        return elapsed

    def close(self):
        self.region.close()


class NmsTop:
    """
    Driver for nms_top_0.
    Two separate AXI-Lite interfaces:
      ctrl    (s_axi_CTRL)    -- AP_CTRL, thresholds, num_out
      control (s_axi_control) -- boxes_in / boxes_out buffer pointers (the fix)
    """

    def __init__(self):
        self.ctrl = AxiLiteRegion(NMS_CTRL_BASE)
        self.control = AxiLiteRegion(NMS_CONTROL_BASE)

    def run(self, boxes_in_addr: int, boxes_out_addr: int,
            conf_thresh: float, nms_thresh: float) -> tuple[float, int]:
        # Buffer pointers go through the s_axi_control interface
        self.control.write64_split(
            NMS_CONTROL_BOXES_IN_LO, NMS_CONTROL_BOXES_IN_HI, boxes_in_addr
        )
        self.control.write64_split(
            NMS_CONTROL_BOXES_OUT_LO, NMS_CONTROL_BOXES_OUT_HI, boxes_out_addr
        )

        # Thresholds and AP_START go through s_axi_CTRL
        self.ctrl.write32(NMS_CTRL_CONF_THRESH, to_fixed(conf_thresh))
        self.ctrl.write32(NMS_CTRL_NMS_THRESH, to_fixed(nms_thresh))
        self.ctrl.write32(NMS_CTRL_AP_CTRL, AP_START)

        elapsed = poll_ap_done(self.ctrl, NMS_CTRL_AP_CTRL)
        num_out = self.ctrl.read32(NMS_CTRL_NUM_OUT_DATA)
        return elapsed, num_out

    def close(self):
        self.ctrl.close()
        self.control.close()


def sanity_check():
    """
    Quick liveness check -- reads AP_CTRL from all three interfaces and
    confirms each reports ap_idle (0x4), matching a freshly-loaded,
    not-yet-triggered IP. This is the same check already confirmed working
    live on the board.
    """
    decode = Decode()
    nms = NmsTop()
    try:
        d_ctrl = decode.region.read32(DECODE_AP_CTRL)
        n_ctrl = nms.ctrl.read32(NMS_CTRL_AP_CTRL)
        print(f"decode_0     AP_CTRL @ {hex(DECODE_BASE)}: {hex(d_ctrl)}")
        print(f"nms_top CTRL AP_CTRL @ {hex(NMS_CTRL_BASE)}: {hex(n_ctrl)}")
        for name, val in (("decode_0", d_ctrl), ("nms_top_0", n_ctrl)):
            if val & AP_IDLE:
                print(f"  {name}: ap_idle set -- ready to trigger")
            else:
                print(f"  {name}: WARNING -- ap_idle not set, unexpected state")
    finally:
        decode.close()
        nms.close()


def bridge_test(drm_path: str, dpu_output_int8_path: str,
                 conf_thresh: float = 0.25, nms_thresh: float = 0.45):
    """
    Real end-to-end bridge: takes captured, real DPU output (int8, shape
    (1, 8400, 7)) and runs it through decode_0 -> nms_top_0 on actual
    hardware, using zocl CMA buffers for every intermediate tensor.
    """
    N_BOXES, N_IN, N_OUT_DECODE = 8400, 7, 6
    MAX_DET = 100

    raw = open(dpu_output_int8_path, "rb").read()
    assert len(raw) == N_BOXES * N_IN, (
        f"expected {N_BOXES * N_IN} bytes, got {len(raw)} -- "
        f"check dpu_output_int8_path matches the int8 dump"
    )

    dpu_out_buf   = ZoclBuffer(drm_path, N_BOXES * N_IN)             # decode_0 input
    decode_out_buf = ZoclBuffer(drm_path, N_BOXES * N_OUT_DECODE * 4) # decode_0 output (ap_fixed<20,12> -> stored as 4B each)
    nms_out_buf    = ZoclBuffer(drm_path, MAX_DET * 6 * 4)            # nms_top_0 output

    dpu_out_buf.write(raw)

    decode = Decode()
    nms = NmsTop()
    try:
        print(f"decode_0 input  @ phys {hex(dpu_out_buf.phys_addr)}")
        print(f"decode_0 output @ phys {hex(decode_out_buf.phys_addr)}")
        t_decode = decode.run(dpu_out_buf.phys_addr, decode_out_buf.phys_addr)
        print(f"decode_0 done in {t_decode*1000:.3f} ms")

        print(f"nms_top_0 input  @ phys {hex(decode_out_buf.phys_addr)}")
        print(f"nms_top_0 output @ phys {hex(nms_out_buf.phys_addr)}")
        t_nms, num_out = nms.run(
            decode_out_buf.phys_addr, nms_out_buf.phys_addr,
            conf_thresh, nms_thresh
        )
        print(f"nms_top_0 done in {t_nms*1000:.3f} ms, num_out={num_out}")
        print(f"total decode+NMS latency: {(t_decode + t_nms)*1000:.3f} ms")

        out_raw = nms_out_buf.read(num_out * 6 * 4)
        vals = struct.unpack(f"<{num_out * 6}f", out_raw)
        for i in range(num_out):
            x1, y1, x2, y2, score, cls = vals[i*6:(i+1)*6]
            cls_name = "fire" if cls > 0.5 else "smoke"
            print(f"  det[{i}] {cls_name} score={score:.3f} box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
    finally:
        decode.close()
        nms.close()
        dpu_out_buf.close()
        decode_out_buf.close()
        nms_out_buf.close()


if __name__ == "__main__":
    import sys
    if os.geteuid() != 0:
        raise SystemExit("This driver needs root (mmaps /dev/mem). Run with sudo.")

    if len(sys.argv) > 1 and sys.argv[1] == "bridge":
        drm_path = sys.argv[2] if len(sys.argv) > 2 else "/dev/dri/renderD129"
        dpu_bin  = sys.argv[3] if len(sys.argv) > 3 else "/home/root/dpu_output_int8.bin"
        bridge_test(drm_path, dpu_bin)
    else:
        sanity_check()
