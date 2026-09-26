#!/bin/sh
# Sweep worker counts to find where frame-level pipelining saturates.
# Expect throughput to rise until it hits the DPU's serial limit (~43 ms/frame,
# ~23 FPS) and flatten there, while per-frame latency grows -- that trade is
# the whole point of pipelining and should be visible in the numbers.
cd /home/root/hybrid_pkg
export XLNX_VART_FIRMWARE=/lib/firmware/xilinx/kv260-hybrid2/hybrid.xclbin
export XRT_INI_PATH=/home/root/xrt.ini
for w in 1 2 3 4; do
    echo "########## workers = $w ##########"
    timeout 600 python3 -u pipelined_throughput.py --workers $w --frames 40 2>&1 \
        | grep -E "workers|frames|throughput|latency|speed-up"
    echo
done
