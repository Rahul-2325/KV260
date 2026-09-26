#!/bin/sh
# try_polling_mode.sh -- run ON THE BOARD
#
# The DPU is alive (registers 0x1F0/0x1F4 report fingerprint
# 0x0101000016010407 correctly, timestamp reg non-zero) but never
# executes: VART reports `dpu timeout! core_idx = 0` with all profiling
# counters at zero, and the command registers 0x40..0x98 stay zero, so
# the start command is not reaching the DPU.
#
# This test switches the CU from interrupt-driven to POLLED completion
# (m_int_enable 1 -> 0). If the problem is interrupt routing (our DPU
# IRQ lands on pl_ps_irq0[0] via a 1-port concat, which may not match
# what zocl's CU interrupt controller expects), polling sidesteps it.
set -e
cd /home/root/xclbin_build

sed -i 's/"m_int_enable": "1"/"m_int_enable": "0"/' ip.json
echo "--- ip.json ---"
grep -E 'int_enable|interrupt_id|base_address' ip.json

xclbinutil \
  --add-section EMBEDDED_METADATA:RAW:emb.xml \
  --add-section IP_LAYOUT:JSON:ip.json \
  --add-section MEM_TOPOLOGY:JSON:mem.json \
  --add-section CONNECTIVITY:JSON:conn.json \
  --output kv260_hybrid.xclbin --force >/dev/null 2>&1
cp kv260_hybrid.xclbin /lib/firmware/xilinx/kv260-hybrid/
echo "xclbin rebuilt in POLLING mode"

cd /home/root
xmutil unloadapp >/dev/null 2>&1 || true
sleep 2
xmutil loadapp kv260-hybrid >/dev/null 2>&1
sleep 3

XLNX_VART_FIRMWARE=/lib/firmware/xilinx/kv260-hybrid/kv260_hybrid.xclbin \
  timeout 120 python3 -u e2e_graphrunner.py \
    --xmodel /home/root/lcam_v5_2p5.xmodel --image WEB09971.jpg --runs 3 2>&1 | tail -16
