import mmap, os, struct
CTRL_BASE   = 0x80070000
CTRL_R_BASE = 0x80100000
def rw(base, off, val=None):
    fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
    mm = mmap.mmap(fd, 0x1000, mmap.MAP_SHARED, mmap.PROT_READ|mmap.PROT_WRITE, offset=base)
    if val is not None:
        struct.pack_into('<I', mm, off, val & 0xFFFFFFFF)
    out = struct.unpack_from('<I', mm, off)[0]
    mm.close(); os.close(fd)
    return out
print("ap_ctrl before      = 0x%08x  (expect 0x4 = IDLE)" % rw(CTRL_BASE, 0x00))
print("--- writing scalars H=4 W=4 C=2 fp_in=0 fp_out=0 ---")
for name, off, v in [("H",0x10,4), ("W",0x18,4), ("C",0x20,2), ("fp_in",0x28,0), ("fp_out",0x30,0)]:
    got = rw(CTRL_BASE, off, v)
    flag = "OK" if got == v else "*** MISMATCH ***"
    print("  %-7s off=0x%02x wrote=%-4d readback=%-10d %s" % (name, off, v, got, flag))
print("--- writing pointers (control_r) ---")
for name, off, v in [("feat_in",0x10,0x3258000), ("feat_out",0x1c,0x477a000)]:
    rw(CTRL_R_BASE, off, v)
    rw(CTRL_R_BASE, off+4, 0)
    lo = rw(CTRL_R_BASE, off); hi = rw(CTRL_R_BASE, off+4)
    flag = "OK" if lo == v else "*** MISMATCH ***"
    print("  %-8s off=0x%02x wrote_lo=0x%08x readback_lo=0x%08x hi=0x%08x %s" % (name, off, v, lo, hi, flag))
print("ap_ctrl after config = 0x%08x" % rw(CTRL_BASE, 0x00))
