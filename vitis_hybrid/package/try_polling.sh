#!/bin/sh
# try_polling.sh -- attempt to run the DPU with XRT in polling mode.
#
# The DPU completes but its interrupt never reaches zocl (see
# probe_dpu_hang.py). If KDS polls the CU status register instead of waiting
# on an IRQ, the missing interrupt stops mattering.
echo "=== all Runtime.* ini keys XRT recognises ==="
strings /usr/lib/libxrt_core.so.2 2>/dev/null | grep -E "^Runtime\." | sort -u
echo
echo "=== reset the PL so the DPU starts from idle ==="
xmutil unloadapp >/dev/null 2>&1; sleep 2
xmutil loadapp kv260-hybrid2 >/dev/null 2>&1; sleep 3

cat > /home/root/xrt.ini <<'INI'
[Runtime]
ert_polling=true
INI
echo "--- /home/root/xrt.ini ---"
cat /home/root/xrt.ini

echo
echo "=== run the DPU with polling enabled ==="
cd /home/root/hybrid_pkg
XRT_INI_PATH=/home/root/xrt.ini XLNX_DPU_TIMEOUT=30000 \
    timeout 120 python3 _dpu_child.py 2>&1 | tail -20
echo
echo "=== interrupt counters (still expected 0 if polling did the work) ==="
grep zocl_irq_intc /proc/interrupts
echo DONE
