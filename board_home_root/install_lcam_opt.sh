#!/bin/sh
# install_lcam_opt.sh -- run ON THE BOARD
# Installs + loads the standalone OPTIMISED lcam_attention_gate bitstream,
# then does a SAFE read-only register probe before any DMA is attempted.
set -e
D=/lib/firmware/xilinx/kv260-lcamopt
P=/home/root/lcam_opt_pkg

cd $P
echo "=== compiling overlay ==="
dtc -@ -O dtb -o lcam_opt.dtbo pl_overlay_lcam_opt.dts 2>&1 | grep -v Warning || true
ls -la lcam_opt.dtbo

echo "=== installing ==="
mkdir -p $D
cp lcam_opt.bit.bin lcam_opt.dtbo $D/
printf '{ "shell_type" : "XRT_FLAT", "num_slots": "1" }' > $D/shell.json
ls -la $D

echo "=== loading ==="
xmutil unloadapp >/dev/null 2>&1 || true
sleep 2
xmutil loadapp kv260-lcamopt
sleep 3

echo "=== /dev/dri ==="
ls /dev/dri/

echo "=== SAFE register probe (read-only) ==="
python3 - <<'PY'
import mmap, os, struct
def rd(base, off=0):
    fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
    mm = mmap.mmap(fd, 0x1000, mmap.MAP_SHARED, mmap.PROT_READ, offset=base)
    v = struct.unpack_from('<I', mm, off)[0]
    mm.close(); os.close(fd)
    return v
print("  ctrl   @0x80060000 ap_ctrl = 0x%08x  (expect 0x4 = IDLE)" % rd(0x80060000))
print("  ctrl_r @0x800F0000 [0x00]  = 0x%08x" % rd(0x800F0000))
PY
echo "DONE"
