#!/bin/sh
# build_hybrid_xclbin.sh -- run ON THE BOARD
#
# Builds a VART-loadable xclbin for the hybrid design (DPU + LCAM IP).
#
# A hand-made xclbin containing only IP_LAYOUT + MEM_TOPOLOGY is REJECTED:
#     [XRT] ERROR: No xml meta data in xclbin
#     [UNILOG][FATAL][VART_LOAD_XCLBIN_FAIL][Bitstream download failed!]
# XRT also needs EMBEDDED_METADATA (an XML describing the kernel, its
# ports/args and -- critically -- the instance addrRemap base).
#
# Strategy: take EVERY section from the known-good stock benchmark xclbin
# and change ONLY what differs in our design:
#     DPU base address 0xA001_0000  ->  0x8F00_0000   (IP_LAYOUT + XML)
#     kernel clocks    600/300 MHz  ->  549.945/274.9725 MHz (cosmetic)
# MEM_TOPOLOGY and CONNECTIVITY are reused verbatim: every bank is just
# "MEM_DRAM, 2GB, base 0x0", so the tag names are immaterial, and reusing
# them keeps the arg->bank indices in CONNECTIVITY valid.

set -e
STOCK=/lib/firmware/xilinx/kv260-benchmark-b4096/kv260-benchmark-b4096.xclbin
W=/home/root/xclbin_build
OUT=/lib/firmware/xilinx/kv260-hybrid/kv260_hybrid.xclbin

rm -rf $W; mkdir -p $W; cd $W

echo "=== dumping stock sections ==="
xclbinutil --input $STOCK --dump-section EMBEDDED_METADATA:RAW:emb.xml       --force >/dev/null
xclbinutil --input $STOCK --dump-section IP_LAYOUT:JSON:ip.json              --force >/dev/null
xclbinutil --input $STOCK --dump-section MEM_TOPOLOGY:JSON:mem.json          --force >/dev/null
xclbinutil --input $STOCK --dump-section CONNECTIVITY:JSON:conn.json         --force >/dev/null
ls -la

echo "=== patching DPU base address 0xA0010000 -> 0x8F000000 ==="
sed -i 's/0x00A0010000/0x008F000000/g'  emb.xml
sed -i 's/0xa0010000/0x8f000000/gI'     ip.json

# --- interrupt id -------------------------------------------------
# The stock xclbin declares m_interrupt_id = 1, matching ITS wiring. In
# OUR design dpu_concat_irq has NUM_PORTS=1 and its In0 is hier_dpu/INTR,
# so the DPU's interrupt lands on pl_ps_irq0[0] == zocl interrupt index 0
# (zocl owns SPI 0x59..0x60 = pl_ps_irq0[7:0]).
# Leaving it at 1 makes zocl wait on a line that never fires -> the CU
# never reports completion and VART aborts with:
#     cu timeout! ... state 1
#     dpu timeout! core_idx = 0
#     LSTART 0 LEND 0 CSTART 0 ... (all counters zero = never ran)
sed -i 's/"m_interrupt_id": "1"/"m_interrupt_id": "0"/' ip.json
# clock metadata: our clk_wiz produces 549.945 / 274.9725 MHz
sed -i 's/599\.994000MHz/549.945000MHz/' emb.xml
sed -i 's/299\.997000MHz/274.972500MHz/' emb.xml

echo "--- verify patch ---"
grep -o 'addrRemap base="[^"]*"' emb.xml
grep -o '"m_base_address": "[^"]*"' ip.json | head -1
grep -o 'frequency="[^"]*"' emb.xml

echo "=== assembling xclbin ==="
xclbinutil \
  --add-section EMBEDDED_METADATA:RAW:emb.xml \
  --add-section IP_LAYOUT:JSON:ip.json \
  --add-section MEM_TOPOLOGY:JSON:mem.json \
  --add-section CONNECTIVITY:JSON:conn.json \
  --output kv260_hybrid.xclbin --force

echo "=== installing ==="
cp kv260_hybrid.xclbin $OUT
xclbinutil --info --input $OUT 2>/dev/null | grep -E "Kernels|Sections" || true
echo "DONE: $OUT"
