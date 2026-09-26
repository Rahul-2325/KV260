# `vitis_hybrid/` — Option B: DPU + LCAM in one xclbin, via the Vitis link flow

New folder, added 2026-08-30. Nothing in the rest of the package was
modified. Full narrative is `PROJECT_HISTORY.md` §31.

## What this is

The hybrid design the project has wanted since Day 1: **DPUCZDX8G B4096 and
the optimised `lcam_attention_gate` running in the same bitstream**, so the
DPU handles the 111 conv layers and the one operator it cannot express
natively — the LCAM attention gate, which otherwise falls back to the ARM
CPU at 83.2 ms — runs in the PL beside it.

**Status: built, timing-clean, and HARDWARE-VERIFIED** (board app
`kv260-hybrid2`). Both CUs run; the full model runs end-to-end.

```
hybrid.xclbin   WNS +0.031 ns   TNS 0.000   0 failing endpoints / 362093
                "All user specified timing constraints are met."
CUs:  DPUCZDX8G_1 @ 0xa0010000    lcam_attention_gate_opt_1 @ 0xa0020000
Device: LUT 52.85%  slices 97.00%  BRAM 68.40%  URAM 71.88%  DSP 57.37%

MEASURED ON HARDWARE (see PROJECT_HISTORY 32/33)
  LCAM gate, four layers   :   1.82 ms   bit-exact   (CPU: 200.30 ms)
  DPU, 8 subgraphs         :  41.15 ms
  full model end-to-end    : 426.51 ms   2.34 FPS
  projected with the IP    : 228.03 ms   4.39 FPS    (1.87x)
```
The 1.87x is a projection: it substitutes the measured 1.82 ms for the
measured 200.30 ms, but the real DPU->IP->DPU pipeline is not built yet,
so the hand-off cost is not included.

## Using it on the board -- TWO THINGS ARE REQUIRED

1. `XRT_INI_PATH=/home/root/xrt.ini` containing
   ```
   [Runtime]
   ert_polling=true
   ```
   Without this EVERY DPU call times out after 10 s and leaves the CU
   wedged. v++ cascades all CU interrupts through `axi_intc_0` into one PS
   line, but zocl waits on a separate GIC line per CU and nothing programs
   the intc -- so no interrupt is ever delivered. Polling sidesteps it.
   The DPU itself is fine: it completes in 1.41 s wall with real profiling
   counters (PROJECT_HISTORY 33.2).

2. `lcam_hybrid.xmodel` -- the only model matching this DPU's fingerprint
   `0x101000012010407`. `lcam_v5.xmodel` (…56…) and `lcam_v5_2p5.xmodel`
   (…16…) are both rejected by VART's fingerprint check.

After any DPU timeout: `xmutil unloadapp; xmutil loadapp kv260-hybrid2`,
and confirm the CU reads ap_ctrl 0x4 before retrying.

## Why not the Vivado block design we already had

`kv260_hybrid_v2` (§29.13) produced a working *bitstream*, but XRT could
never start the DPU in it. A Vivado-TRD DPU is a raw IP, so XRT registered
it from IP_LAYOUT and drove it with the generic AP_CTRL_HS handshake —
writing `ap_start` into what is actually a read-only DPU **version**
register, and hanging CU(0). That is a contract mismatch, not a metadata
bug, which is why cloning xclbin sections never fixed it.

Linking through `v++` packages the DPU with its own `kernel.xml`, so
XRT/VART drive it with the protocol it actually implements.

## Why every earlier `v++` attempt failed

Not licensing, not versions. **There was no accelerated platform.**
`platforminfo` on the old `kv260_9ip_v2.xpfm` reports no clocks and no AXI
ports, so `v++` correctly refused it with
`[v++ 60-1606] ... is a non-accelerated platform`. `build_platform.tcl`
here builds a real extensible platform (PFM_NAME / PFM.CLOCK /
PFM.AXI_PORT / PFM.IRQ, `platform.extensible true`) which now publishes
3 clocks and 7 memory ports.

## Build order

```powershell
# 1. extensible platform  -> out/kv260_ext.xsa
vivado -mode batch -source build_platform.tcl

# 2. XSA -> .xpfm
xsct make_xpfm.tcl

# 3. LCAM kernel -> .xo     (needs LCAM_PERIOD env var, see run_xo.tcl)
$env:LCAM_PERIOD="3.333"; vitis_hls -f run_xo.tcl

# 4. link (~56 min)
v++ -l -t hw --platform <...>.xpfm --config hybrid.cfg `
    -o hybrid.xclbin dpu.xo lcam_attention_gate_opt.xo

# 5. package for the board
package\make_package.ps1
```
The DPU `.xo` is reused as-is from
`C:\Xilinx\projects\DPUCZDX8G\prj\Vitis\binary_container_1\dpu.xo`
(B4096 + URAM_ENABLE).

## Two source changes the kernel flow required

`lcam_attention_gate_opt_kernel.cpp` is the Vitis-kernel variant of
`hls_ip_sources/lcam_attention_gate/lcam_attention_gate_opt.cpp`, kept
separate so the original IP-flow source is untouched.

1. **The §23 `control`/`control_r` split is fatal in kernel mode.** In the
   IP flow it was a harmless quirk (and was correctly ruled out as the
   all-zeros cause). Here HLS hard-errors with `[HLS 214-219]`. Fixed by
   pinning the `m_axi` offset registers onto `bundle=control` explicitly.

2. **The 512-bit bus was over-built.** A ZynqMP `S_AXI_HP` port is 128
   bits wide, so a 512-bit kernel port only makes the platform insert a
   width converter while 64 multiply lanes idle 3 of every 4 cycles. The
   old standalone measurement proves it: 6.5 MB over 128 bits @ 100 MHz
   (1.6 GB/s) predicts 4.0 ms, and §30.4 measured 3.50 ms — the bus, not
   the arithmetic, set the time. Narrowing to 128 bits cost no throughput
   and shrank the placed kernel from 28993 to 10062 LUT, which is what
   turned WNS −0.823 ns into +0.031 ns.

## A third trap: the kernel flow uses ap_ctrl_chain

The nine IP-flow cores are `ap_ctrl_hs`, where reading `ap_done` clears it.
A v++-linked kernel is `ap_ctrl_chain`: **`ap_done` is sticky until you
write `ap_continue` (bit 4)**. Reusing `hw_ip_driver.poll_ap_done()` makes
the FIRST run look perfect and every later run return in ~0.03 ms against
an untouched output buffer, then wedges the CU at `ap_ctrl = 0x203`.
Correct sequence, as in `package/bench_lcam_hybrid.py`:
```
wait ap_idle -> write args -> ap_start(0x1) -> poll ap_done(0x2)
             -> write ap_continue(0x10)
```
If the other eight IPs are ever moved to the kernel flow, their driver
needs the same change.

## Deploying (next session)

Copy `package/out/` to the board and run `install_hybrid.sh`. Then, in
order, and **do not skip 2 or 4**:

2. `xdputil query` — read the DPU fingerprint. Enabling URAM moved it once
   already (§29.15), and it decides whether `lcam_v5.xmodel` or
   `lcam_v5_2p5.xmodel` will load.
4. The LCAM CU now lives at **0xa0020000**, not the old 0x80060000.
   `hw_ip_driver.py`'s `IP_ADDR` must be repointed before any register
   poke, or it writes into the DPU's aperture instead.
