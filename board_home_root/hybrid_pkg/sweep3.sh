cd /home/root/hybrid_pkg
export XLNX_VART_FIRMWARE=/lib/firmware/xilinx/kv260-hybrid3/hybrid3.xclbin
export XRT_INI_PATH=/home/root/xrt.ini
for w in 1 2 4; do echo "### workers=$w"; timeout 600 python3 -u pipelined_throughput.py --workers $w --frames 40 2>&1 | grep -E 'throughput|latency'; done
