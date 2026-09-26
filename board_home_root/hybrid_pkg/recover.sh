#!/bin/sh
# Recover the board after a wedged CU: kill the spinning client, reload the
# PL through xmutil (the only safe reset path), and report the state.
pkill -9 -f hybrid_pipeline.py 2>/dev/null
pkill -9 -f measure_power.py 2>/dev/null
sleep 2
echo "--- load after kill ---"
uptime
echo "--- kernel messages ---"
dmesg | tail -10
echo "--- reload PL ---"
xmutil unloadapp >/dev/null 2>&1
sleep 2
xmutil loadapp kv260-hybrid3
sleep 3
echo "--- CU ap_ctrl (expect 0x4 IDLE) ---"
python3 - <<'PY'
import mmap, os, struct
fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
mm = mmap.mmap(fd, 0x1000, mmap.MAP_SHARED, mmap.PROT_READ, offset=0xa0020000)
print("  lcam CU = 0x%x" % struct.unpack_from('<I', mm, 0)[0])
mm.close(); os.close(fd)
fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
mm = mmap.mmap(fd, 0x1000, mmap.MAP_SHARED, mmap.PROT_READ, offset=0xa0010000)
print("  DPU     = 0x%x" % struct.unpack_from('<I', mm, 0)[0])
mm.close(); os.close(fd)
PY
echo DONE
