#!/bin/sh
# reset_and_probe.sh -- reload the PL to clear a stuck CU, then report state.
#
# Reloading through xmutil re-programs the fabric AND keeps the device tree,
# clocks and zocl registration consistent, which is why it is the only safe
# way to reset. (JTAG re-programming leaves those stale and hangs the
# interconnect on the next AXI-Lite access -- five board crashes, section 25.)
xmutil unloadapp >/dev/null 2>&1
sleep 2
xmutil loadapp kv260-hybrid2
sleep 3

echo "--- CU ap_ctrl after reload (expect 0x4 = IDLE) ---"
python3 - <<'PY'
import mmap, os, struct
fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
mm = mmap.mmap(fd, 0x1000, mmap.MAP_SHARED, mmap.PROT_READ, offset=0xa0020000)
print("  lcam CU 0xa0020000 ap_ctrl = 0x%x" % struct.unpack_from('<I', mm, 0)[0])
mm.close(); os.close(fd)
PY

echo "--- how the proven 9-IP driver drives ap_ctrl ---"
grep -n "ap_start\|ap_done\|ap_idle\|def run\|def wait\|while " /home/root/hw_ip_driver.py | head -40
echo "--- and how ip_wrappers calls it ---"
grep -n "def run\|0x00\|start\|poll" /home/root/ip_wrappers.py | head -25
