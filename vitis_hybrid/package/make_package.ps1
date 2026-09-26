# make_package.ps1
# Turns the v++ link output (hybrid.xclbin) into the four files the KV260's
# xmutil app loader expects in /lib/firmware/xilinx/kv260-hybrid/.
#
# Run AFTER the v++ link finishes. Produces package/out/:
#   hybrid.bit.bin  - the PL bitstream, bootgen-wrapped for fpga_manager
#   hybrid.dtbo     - compiled overlay (built on the board; dtc is there)
#   hybrid.xclbin   - the XRT metadata VART loads to find the DPU + LCAM CUs
#   shell.json      - tells xmutil this is a flat XRT design
#
# Why both a .bit.bin AND an .xclbin: xmutil/fpga_manager programs the
# fabric from the .bit.bin named in the overlay's firmware-name, while
# XRT/VART separately reads the .xclbin for IP_LAYOUT, MEM_TOPOLOGY and
# the DPU's kernel metadata. The stock kv260-dpu app ships exactly this
# pair, and /etc/vart.conf points at the xclbin.

$ErrorActionPreference = "Stop"
$H   = "C:\Xilinx\projects\kv260_hybrid_vitis"
$OUT = "$H\package\out"
$XCL = "$H\hybrid.xclbin"

if (-not (Test-Path $XCL)) { throw "hybrid.xclbin not found -- the v++ link has not produced it yet" }
New-Item -ItemType Directory -Force -Path $OUT | Out-Null

$xclbinutil = "C:\Xilinx\Vitis\2022.2\bin\xclbinutil.bat"
$bootgen    = "C:\Xilinx\Vitis\2022.2\bin\bootgen.bat"

# NOTE: in 2022.2 xclbinutil --info prints to stdout and exits 255 even on
# success. Do NOT pass --output alongside it -- that is the xclbin output
# path, so it writes a 7.8 MB copy of the xclbin instead of a text report.
Write-Host "=== compute units in the xclbin (expect exactly two) ==="
$info = & $xclbinutil --input $XCL --info 2>&1
$info | Select-String -Pattern "Instance:|Base Address: 0xa" | ForEach-Object { "  " + $_.Line.Trim() }
$info | Set-Content -Encoding ascii "$OUT\xclbin_info.txt"

Write-Host "`n=== extracting raw bitstream ==="
& $xclbinutil --input $XCL --dump-section "BITSTREAM:RAW:$OUT\hybrid.bit" --force

# bootgen needs a .bif; -process_bitstream bin emits <name>.bit.bin next to
# the input .bit, which is the format zynqmp fpga_manager expects.
$bif = "$OUT\hybrid.bif"
@"
all:
{
    $($OUT -replace '\\','/')/hybrid.bit
}
"@ | Set-Content -Encoding ascii $bif

Write-Host "`n=== bootgen -> hybrid.bit.bin ==="
Push-Location $OUT
& $bootgen -image $bif -arch zynqmp -process_bitstream bin -w
Pop-Location

Copy-Item $XCL "$OUT\hybrid.xclbin" -Force
'{ "shell_type" : "XRT_FLAT", "num_slots": "1" }' | Set-Content -Encoding ascii "$OUT\shell.json"
Copy-Item "$H\package\pl_overlay_hybrid.dts" "$OUT\" -Force
Copy-Item "$H\package\install_hybrid.sh"     "$OUT\" -Force -ErrorAction SilentlyContinue

Write-Host "`n=== package contents ==="
Get-ChildItem $OUT | Select-Object Name, Length
