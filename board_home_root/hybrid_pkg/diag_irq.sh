#!/bin/sh
# diag_irq.sh -- is the DPU's completion interrupt actually reaching zocl?
#
# Theory under test: v++ cascades every CU interrupt through axi_intc_0 into
# ONE PS line (pl_ps_irq0 = IRQ 89), but zocl assigns each CU its own IRQ
# index from the overlay's list (CU0 -> 89, CU1 -> 90, ...). The DPU is CU
# index 1, so its done-interrupt would arrive on 89 and nobody would be
# waiting on it -- which looks exactly like "CU accepted the command, then
# timed out with is_done 0".
#
# If that is right we should see: zocl holding several IRQs, and a nonzero
# count ONLY on the first one.
echo "=== zocl / PL interrupts and their counts ==="
grep -iE "zocl|zyxclmm|xilinx" /proc/interrupts || echo "  (no zocl lines)"
echo
echo "=== all PL-range IRQs (89..96 = 0x59..0x60) ==="
awk 'NR==1 || /^ *(89|90|91|92|93|94|95|96):/' /proc/interrupts
echo
echo "=== zocl kernel messages ==="
dmesg | grep -iE "zocl|xclbin|kds|cu_" | tail -25
echo
echo "=== XRT version ==="
xbutil --version 2>/dev/null | head -8
echo
echo "=== existing xrt.ini, if any ==="
for f in /etc/xrt.ini /home/root/xrt.ini /home/root/hybrid_pkg/xrt.ini; do
    [ -f "$f" ] && echo "--- $f" && cat "$f"
done
echo "DONE"
