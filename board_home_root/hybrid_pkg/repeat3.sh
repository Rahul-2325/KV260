cd /home/root/hybrid_pkg
export XLNX_VART_FIRMWARE=/lib/firmware/xilinx/kv260-hybrid3/hybrid3.xclbin
export XRT_INI_PATH=/home/root/xrt.ini
for i in 1 2 3; do timeout 300 python3 -u pipelined_throughput.py --workers 2 --frames 40 2>&1 | grep throughput; done
echo '--- single-frame latency, 3 repeats ---'
for i in 1 2 3; do timeout 300 python3 hybrid_pipeline.py --gates ip --round ref --runs 5 2>&1 | grep 'mean total'; done
