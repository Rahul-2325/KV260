#!/bin/bash
# ============================================================
# KV260 Deployment Script
# Run this on your WSL/Linux machine to copy files to board
# ============================================================

# ── Configuration — edit these ────────────────────────────────
BOARD_IP="192.168.1.100"          # replace with your board's IP
BOARD_USER="root"
BOARD_DIR="/home/root/lcam_deploy"

XCLBIN="/mnt/c/Xilinx/kv260_lcam.xclbin"          # original working xclbin
XMODEL="$HOME/wildfire_project/compiled_v5/lcam_v5.xmodel"
META="$HOME/wildfire_project/compiled_v5/meta.json"
TEST_IMG="$HOME/wildfire_project/float_result.jpg"
INFER_SCRIPT="./infer_kv260.py"

# ── Step 1: Verify all files exist locally ─────────────────────
echo "=== Checking local files ==="
for f in "$XCLBIN" "$XMODEL" "$META" "$TEST_IMG" "$INFER_SCRIPT"; do
    if [ -f "$f" ]; then
        echo "  OK: $f"
    else
        echo "  MISSING: $f"
    fi
done

# ── Step 2: Copy to board ──────────────────────────────────────
echo ""
echo "=== Copying files to board $BOARD_IP ==="
ssh ${BOARD_USER}@${BOARD_IP} "mkdir -p ${BOARD_DIR}"

scp "$XCLBIN"      ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/kv260_lcam.xclbin
scp "$XMODEL"      ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/lcam_v5.xmodel
scp "$META"        ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/meta.json
scp "$TEST_IMG"    ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/test.jpg
scp "$INFER_SCRIPT" ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/infer_kv260.py

echo ""
echo "=== Files on board ==="
ssh ${BOARD_USER}@${BOARD_IP} "ls -la ${BOARD_DIR}/"

echo ""
echo "=== To run inference on board ==="
echo "ssh ${BOARD_USER}@${BOARD_IP}"
echo "cd ${BOARD_DIR}"
echo "python3 infer_kv260.py --xclbin kv260_lcam.xclbin --xmodel lcam_v5.xmodel --image test.jpg"
