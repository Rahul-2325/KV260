#!/bin/sh
# probe_polling.sh -- find a way to make zocl/KDS poll the CU instead of
# waiting for an interrupt that our platform never delivers.
#
# Established by probe_dpu_hang.py: the DPU DOES complete (ap_ctrl 0x4 -> 0x6,
# real profiling counters, 2071580 cycles) but no interrupt ever reaches
# zocl, because v++ cascades every CU interrupt through axi_intc_0 into a
# single PS line that nothing programs, while zocl waits on two direct GIC
# lines (121/122). Polling sidesteps the whole issue.
echo "=== zocl module parameters ==="
ls /sys/module/zocl/parameters/ 2>/dev/null
for f in /sys/module/zocl/parameters/*; do
    [ -f "$f" ] && echo "  $(basename $f) = $(cat $f 2>/dev/null)"
done
echo
echo "=== zocl driver info ==="
modinfo zocl 2>/dev/null | head -30
echo
echo "=== does this xclbin mark CUs as interrupt-capable? ==="
xclbinutil --input /lib/firmware/xilinx/kv260-hybrid2/hybrid.xclbin --info 2>/dev/null \
    | grep -iE "interrupt|IP_LAYOUT|Instance|Base Address" | head -20
echo
echo "=== XRT ini keys recognised (grep the libs) ==="
strings /usr/lib/libxrt_core.so.2 2>/dev/null | grep -iE "^polling|cu_interrupt|kds_polling|ert_polling" | sort -u | head -20
echo DONE
