#!/bin/sh
# build_hybrid_xclbin_full.sh -- run ON THE BOARD
#
# Previous attempt built an xclbin with only 4 sections
#   EMBEDDED_METADATA, IP_LAYOUT, MEM_TOPOLOGY, CONNECTIVITY
# XRT accepted it (no more "No xml meta data"), and the DPU was correctly
# identified (xdputil reported DPU Arch ..._0101000016010407, cu_addr
# 0x8f000000), but every run ended in
#     cu timeout! ... ToCU: 0us ... dpu timeout! core_idx = 0
# with all DPU profiling counters zero -- i.e. zocl never dispatched the
# command to the CU. Changing m_interrupt_id (1->0) and m_int_enable
# (1->0, polling) made no difference.
#
# The stock xclbin carries EIGHT sections:
#   MEM_TOPOLOGY, IP_LAYOUT, CONNECTIVITY, BUILD_METADATA,
#   EMBEDDED_METADATA, SYSTEM_METADATA, GROUP_CONNECTIVITY, GROUP_TOPOLOGY
# The GROUP_* sections are what modern XRT actually consults for memory
# grouping / CU association at dispatch time; omitting them is the most
# likely reason the command never reaches the CU.
#
# So: clone the stock xclbin WHOLESALE and patch only the DPU address.
set -e
STOCK=/lib/firmware/xilinx/kv260-benchmark-b4096/kv260-benchmark-b4096.xclbin
W=/home/root/xclbin_full
OUT=/lib/firmware/xilinx/kv260-hybrid/kv260_hybrid.xclbin

rm -rf $W; mkdir -p $W; cd $W

echo "=== dumping ALL sections from stock ==="
for S in MEM_TOPOLOGY IP_LAYOUT CONNECTIVITY GROUP_TOPOLOGY GROUP_CONNECTIVITY; do
    xclbinutil --input $STOCK --dump-section $S:JSON:$S.json --force >/dev/null 2>&1 \
        && echo "  $S -> json" || echo "  $S MISSING"
done
for S in EMBEDDED_METADATA BUILD_METADATA SYSTEM_METADATA; do
    xclbinutil --input $STOCK --dump-section $S:RAW:$S.raw --force >/dev/null 2>&1 \
        && echo "  $S -> raw" || echo "  $S MISSING"
done

echo "=== patching DPU base 0xA0010000 -> 0x8F000000 ==="
sed -i 's/0x00A0010000/0x008F000000/g' EMBEDDED_METADATA.raw
sed -i 's/0xa0010000/0x8f000000/gI'    IP_LAYOUT.json
sed -i 's/599\.994000MHz/549.945000MHz/' EMBEDDED_METADATA.raw
sed -i 's/299\.997000MHz/274.972500MHz/' EMBEDDED_METADATA.raw
grep -o 'addrRemap base="[^"]*"' EMBEDDED_METADATA.raw
grep -o '"m_base_address": "[^"]*"' IP_LAYOUT.json | head -1

echo "=== assembling (all available sections) ==="
ARGS=""
for S in MEM_TOPOLOGY IP_LAYOUT CONNECTIVITY GROUP_TOPOLOGY GROUP_CONNECTIVITY; do
    [ -f $S.json ] && ARGS="$ARGS --add-section $S:JSON:$S.json"
done
for S in EMBEDDED_METADATA BUILD_METADATA SYSTEM_METADATA; do
    [ -f $S.raw ] && ARGS="$ARGS --add-section $S:RAW:$S.raw"
done
echo "sections: $ARGS"
xclbinutil $ARGS --output kv260_hybrid.xclbin --force 2>&1 | tail -4

cp kv260_hybrid.xclbin $OUT
echo "=== installed; sections now: ==="
xclbinutil --info --input $OUT 2>/dev/null | grep -A3 "Sections:"
echo "DONE"
