#!/bin/sh
# install_hybrid.sh -- run ON THE BOARD, from the directory holding the
# package files (hybrid3.bit.bin, hybrid3.xclbin, pl_overlay_hybrid3.dts,
# shell.json).
#
# Installs the Vitis-linked hybrid design (DPU B4096 + optimised LCAM gate)
# as an xmutil app and loads it.
#
# ALWAYS load through xmutil. Never JTAG-program this bitstream and then
# poke it from Linux: program_hw_devices rewrites only the PL fabric and
# leaves the device tree, the clocks and zocl's CU registration stale, so
# the first AXI-Lite access hangs the interconnect and reboots the board.
# That cost five board crashes in one session (PROJECT_HISTORY.md 25).
set -e

APP=kv260-hybrid3
D=/lib/firmware/xilinx/$APP

echo "=== compiling overlay ==="
dtc -@ -O dtb -o hybrid3.dtbo pl_overlay_hybrid3.dts 2>&1 | grep -v Warning || true
ls -la hybrid3.dtbo

echo "=== installing to $D ==="
mkdir -p $D
cp hybrid3.bit.bin hybrid3.dtbo hybrid3.xclbin shell.json $D/
ls -la $D

echo "=== loading ==="
xmutil unloadapp >/dev/null 2>&1 || true
sleep 2
xmutil loadapp $APP
sleep 3

echo "=== /dev/dri ==="
ls /dev/dri/

echo "=== point VART at this xclbin ==="
# VART reads /etc/vart.conf to find the DPU. Keep a backup of whatever was
# there so the stock DPU app still works after we unload.
[ -f /etc/vart.conf ] && [ ! -f /etc/vart.conf.orig ] && cp /etc/vart.conf /etc/vart.conf.orig
printf 'firmware: %s/hybrid3.xclbin\n' "$D" > /etc/vart.conf
cat /etc/vart.conf

echo "=== XRT sees these compute units ==="
# Expect TWO: DPUCZDX8G_1 and lcam_attention_gate_opt_1. If the LCAM CU is
# missing the link config did not instantiate it; if the DPU is missing,
# VART will not run at all.
xbutil examine -r all 2>/dev/null | head -60 || true

echo "=== DPU fingerprint (must match the xmodel we run) ==="
# The fingerprint changes with the DPU config -- enabling URAM alone moved
# it from 0x...12 to 0x...16. Whichever value prints here decides which
# xmodel is loadable: lcam_v5.xmodel or lcam_v5_2p5.xmodel.
xdputil query 2>/dev/null | head -40 || true

echo "DONE"

