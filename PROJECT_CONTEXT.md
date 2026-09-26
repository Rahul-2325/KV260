# LCAM-YOLOX Custom FPGA Accelerator Project — Project Context

## What this project is

UAV-based wildfire detection CNN (LCAM-YOLOX) deployed on a Xilinx KV260
(Zynq UltraScale+ MPSoC, xck26-sfvc784-2LV-c). Baseline uses Xilinx's DPU
IP (195 FPS, already working). This project builds 9 **custom** HLS
accelerator IPs to replace the DPU for specific subgraphs, for research
novelty and to natively hardware-accelerate a custom attention mechanism
(LCAM) that DPU can't run natively.

**Read `PROJECT_HISTORY.md` first** — it is the complete, chronological
log of every session, decision, and finding across the whole project.
It's long (many sessions) but the LATE sections (numbered ~15 onward)
are the most relevant/current. Do not skip it; a huge amount of
debugging has already been done and re-deriving it would waste real time.

## Current status (as of last session)

**Design/software side: essentially done and verified.**
- All 9 HLS IPs pass C-simulation (see `hls_ip_sources/`)
- Full Vivado block design (`kv260_custom_noDPU`) builds cleanly to a
  bitstream — synthesis + implementation both complete with 0 errors
- Complete Linux deployment pipeline works: device tree overlay + xclbin
  packaging + `kria-apps-firmware`/`xmutil` loading, all verified working
- A full JTAG + Integrated Logic Analyzer (ILA) debug setup was built
  from scratch and is fully working — can capture live AXI bus signals

**THE ALL-ZEROS BUG IS SOLVED (2026-08-29). ALL 9 IPs NOW PRODUCE CORRECT
OUTPUT ON REAL HARDWARE — 9/9 validated, including the custom LCAM
attention gate.** See `PROJECT_HISTORY.md` §26 for the full story.

Root cause: this board has 4GB DDR in TWO PHYSICALLY DISTINCT regions
(low 2GB at `0x0`, high 2GB at `0x8_0000_0000`) — they are NOT aliases of
each other. The PL's `S_AXI_HP0/1/2_FPD` ports only decode the LOW region.
zocl was returning HIGH-DDR addresses for small allocations, and the
driver "corrected" them by subtracting `0x8_0000_0000` — which pointed the
PL at real but completely unrelated memory. Hence: bus writes succeeded,
ILA addresses "matched", read data looked like random junk, and output was
always zero. The CPU and the PL were never touching the same memory.

The fix (in `board_driver/hw_ip_driver.py`): allocations `>= 64KB` come
from the low-DDR CMA pool, which the PL CAN reach, so `MIN_CMA_ALLOC`
rounds every buffer up to 64KB. The bogus `HPC_ALIAS_BIT` subtraction is
removed and replaced with a loud `MemoryError`. **Never re-introduce that
subtraction** — high and low DDR are different memory, not an alias pair.

Run `debug_scripts/validate_all_9_ips.py` on the board to re-verify.

### What EXISTS vs what does NOT (be precise about this — 2026-08-29)

**WORKS, verified on hardware, reproducible today:**
- All 9 IPs individually produce correct output with synthetic test data.
  Re-verify any time: `debug_scripts/validate_all_9_ips.py` (prints PASS
  per IP; 6 exact-match, 3 hand-verified numerically).
- `debug_scripts/bench_lcam_vs_cpu.py` — the custom LCAM attention gate
  beats CPU numpy int8 at all four real network sizes (2.05x overall).

**DOES NOT EXIST YET — do not claim these:**
- **NO end-to-end inference.** There is no image -> detections pipeline.
  Feeding `WEB09971.jpg` in right now does nothing.
- **NO connected dataflow.** `layer_plan.json` resolves only 24 of 168
  activation inputs; it omits the 21 concat / 20 download / 16 upload /
  12 reshape glue ops. A real engine must be built from the xmodel via
  `xir` (topology IS readable; only the weight blob is not) — NOT from
  layer_plan.json. See §27/§28.
- **NO hybrid bitstream.** The current `kv260-custom` bitstream contains
  the 9 custom IPs and *no DPU*. The recommended hybrid (DPU + LCAM IP)
  needs a Vivado build that does not exist yet — this is the same
  "DPU + lcam_attention_gate" design that was blocked on Day 1.

### *** HEADLINE RESULTS (measured on the HYBRID bitstream) *** — §32/§33
All of the following are measured on `kv260-hybrid2` (DPU B4096 + the
optimised LCAM gate in ONE xclbin, timing met). 2026-08-30.
```
end-to-end (DPU + CPU attention)   : 426.51 ms -> 2.34 FPS
  DPU compute, 8 subgraphs          :  41.15 ms  ( 9.6%)
  CPU fallback                      : 385.36 ms  (90.4%)  <- bottleneck
    of which the 4 LCAM gates       : 200.30 ms  (52%)   <- our IP replaces
    of which the YOLOX head ops     : ~193 ms    (48%)   <- head_* IPs target

custom lcam_attention_gate IN THE HYBRID :   1.82 ms  BIT-EXACT (4 gates)
  -> 110x faster than the same gates on the CPU (200.30 ms)
  ->  22x faster than the ORIGINAL IP (40.60 ms)
  -> 1.92x faster than the standalone optimised build (3.50 ms)
  placed cost: 10062 LUT, 7 BRAM, 0 URAM, 6 DSP

PROJECTED end-to-end with the LCAM IP     : 228.03 ms -> 4.39 FPS (1.87x)
PROJECTED if head_* IPs are added too     : ~50-55 ms -> ~18-20 FPS
```
**The gates are 52% of CPU time, NOT the ~94% this file used to claim.**
That old number was an element-count proxy; §33.4 measures the real per-op
times with `DEEPHI_PROFILING=1`. Use 200.30 ms.

**Careful with CPU baselines** — several appear in the logs, do not mix:
82.1 ms (simple numpy, understates), ~138–166 ms (numpy exact rounding),
83.2 ms (an older partial figure), and **200.30 ms — the measured VART CPU
cost of the four gates, which is the honest denominator for the IP's
speedup — inside a 385.36 ms total CPU fallback.**

The optimised IP lives in
`hls_ip_sources/lcam_attention_gate/lcam_attention_gate_opt.cpp`
(512-bit AXI + integer arithmetic; the float `roundf` had to become
magnitude-rounding-plus-sign or ~2% of outputs differ — proven bit-exact
by `debug_scripts/verify_lcam_int_math.py`). Standalone bitstream:
`C:\Xilinx\projects\lcam_opt_bit\`, board app `kv260-lcamopt`.

### Performance reality (measured, §28) — read before planning
The IPs are CORRECT but SLOW. `conv2d_engine` is a naive loop nest with
no BRAM tiling, no bursts, no unrolling; every operand is fetched
individually over AXI.
```
measured: 16.59 MMAC/s, ~170 ms fixed overhead per layer
network:  13.00 GMAC over 111 conv layers
=> ~13.5 MINUTES per frame  (~158,000x slower than the DPU's 195 FPS)
```
**Replacing the DPU wholesale is not viable.** But the hybrid is a real
win, and it is the project's actual thesis (the DPU *cannot* run LCAM
attention natively — it falls back to the ARM CPU):
```
DPU + CPU attention  : 5.13 + 83.2 = 88.3 ms -> 11.3 FPS
DPU + custom LCAM IP : 5.13 + 40.6 = 45.7 ms -> 21.9 FPS   (~1.9x)
```

### Three ways forward (pick one next session)
- **B — Hybrid (DPU + LCAM IP). *** DONE AND HARDWARE-VERIFIED, §31–§33 ***.**
  Board app **`kv260-hybrid2`**. Both CUs live
  (`DPUCZDX8G_1` @0xa0010000, `lcam_attention_gate_opt_1` @0xa0020000),
  DPU runs through VART, LCAM gate 1.82 ms bit-exact, full model runs
  end-to-end at 426.51 ms. Built with `v++ --link`, not the Vivado block
  design — that is what finally makes XRT drive the DPU correctly.
  **Two things are REQUIRED to use it** (§33.7):
  `XRT_INI_PATH=/home/root/xrt.ini` (`[Runtime] ert_polling=true`), and
  `lcam_hybrid.xmodel` — the only xmodel matching fingerprint
  `0x101000012010407`. Remaining work: build the real DPU→IP→DPU pipeline
  to turn the 1.87x projection into a measurement.
- **A — End-to-end demo on the 9 custom IPs.** Build image -> detections
  using the xmodel graph for topology. Genuine complete demo, but ~13.5
  min/frame. Best if a visual demonstration is needed.
- **C — Write up what exists.** Already a defensible contribution: 9
  working custom IPs on silicon, a hard root-cause bug found and fixed by
  measurement (§26), and a measured 2x speedup on the operator the vendor
  IP cannot express.

Note for A: 7 learned weight tensors were never extracted (§27) and
cannot be synthesised, so full-custom *accuracy* would be degraded.
They do NOT affect B (the gate consumes activations, not weights) and do
not affect any timing/FPS measurement.

## What's been ruled out (historical — all of this was investigated BEFORE the §26 root cause was found)

**These were all correctly eliminated — none of them was the cause. Kept
for the record so they are not re-tested. The actual cause (§26) was never
on this list because everyone reasonably assumed zocl would return an
address the PL could use.**


1. Cache coherency (SYNC_BO ioctl) — verified correct against real Xilinx
   kernel source (`zynq_ioctl.h`), tested both directions, no effect
2. Register offsets — verified against real generated Vitis HLS headers
3. CMA buffer/physical address validity
4. AXI master block-design wiring
5. Missing zocl device tree node — found, fixed
6. Missing xclbin `MEM_TOPOLOGY`/`IP_LAYOUT` — built correctly, no effect
7. Memory bank index ordering in `MEM_TOPOLOGY`
8. **Write address (AWADDR) correctness** — proven bit-exact match via
   live ILA capture vs. the Python-computed physical address
9. **Read address (ARADDR) correctness** — same, proven bit-exact via ILA
10. Full AXI protocol completion (AW/W/B and AR/R channels all complete
    cleanly per ILA — no DECERR/SLVERR, no stuck handshakes)
11. Beat-ordering confusion in captured RDATA
12. Timing race (artificial 100ms delay before triggering the IP — no effect)
13. CPU-side self-consistency — CPU always correctly reads back its own
    writes to the same buffer (100% reliable)
14. Coherent (S_AXI_HPC0_FPD) vs non-coherent (S_AXI_HP0/1/2_FPD) port
    routing — tested both, no difference
15. SMMU/IOMMU address translation — confirmed absent on this system
    (`ls /sys/class/iommu/` is empty)
16. **HLS interface architecture** (`s_axi_control` + auto-split
    `s_axi_control_r` vs. a single unified `s_axi_control` bundle,
    matching the one PROVEN-WORKING reference IP in the project,
    `decode.cpp` from the original DPU build) — this was the strongest,
    best-evidenced hypothesis of the whole investigation, properly
    controlled-tested, and it made NO difference either.

**Critically: the actual DATA on the AXI read channel (RDATA) was
captured once and found to be neither the correct input data NOR zero —
it looked like uninitialized/random memory content.** This was never
fully chased down and may be an important clue (see `PROJECT_HISTORY.md`
section 19).

## ~~Suggested next steps~~ — OBSOLETE, ALL RESOLVED (kept only for context)

**Do NOT work through the list that used to live here.** It was written
before the root cause was known and would send you chasing a solved bug.
For the record, each item is now answered:

1. ~~Compare generated RTL vs `decode.cpp`~~ — unnecessary. The IPs were
   never the problem; they were correct all along. The bug was the
   *address* the driver handed them (§26).
2. ~~Re-run the RDATA capture~~ — the §19 "random-looking RDATA" mystery
   is EXPLAINED: it was real data read from the wrong DRAM region. No
   further capture needed. (An attempt was made anyway and crashed the
   board 5 times — see §24/§25 and gotcha #9 below.)
3. ~~Sanity-check whether any custom AXI master can reach DRAM~~ —
   answered: yes, they all can, once given a PL-reachable address. 9/9
   IPs verified.
4. ~~Build a minimal decode.cpp-style test IP~~ — unnecessary, same
   reason as #1.

**The real open work is now the three options (A/B/C) in "Current status"
above.**

## Hard-won operational gotchas (cost real time — don't rediscover these)

- **NEVER JTAG-program a bitstream and then poke the IPs from Linux.**
  `program_hw_devices` only rewrites the PL fabric; it does not update the
  device tree, clocks, or zocl's IP registration the way `xmutil loadapp`
  does. The first AXI-Lite access then hangs the interconnect and reboots
  the board. This cost 5 board crashes in one session (§25). Use
  `xmutil loadapp kv260-custom`.
- **`pscp` needs `-scp -batch -hostkey ...`** — it does not share plink's
  cached host key, and this board image has no SFTP server.
- **Buffers must be >= 64KB** or zocl returns high-DDR addresses the PL
  cannot reach (§26). The driver now enforces this.
- **Xilinx tools exhaust this 16 GB Windows box at their default
  parallelism, and the errors look like tool bugs (§31.2).** Vivado IP
  generation fails on `ps_e` only with `[Common 17-232] Could not create
  slave interpreter '::ipgen_iptclns'` — fix with
  `set_param general.maxThreads 1`. `v++ --link` dies ~2 min in with
  `boost::filesystem::status: Insufficient system resources` — fix with
  `synth.jobs=2` / `impl.jobs=2` under `[vivado]`. Neither is a licence
  or version problem. Also kill orphaned `vivado.exe` after a failed
  link; five survived one failure holding 5.7 GB.
- **Vitis HLS and Vivado both reject paths containing spaces.** Stage
  sources under `C:\Xilinx\hls_work\...` before building.
- The board's `weight_map.json` is 100% unusable (all 390 entries null) —
  use `board_driver/weight_resolver.py` instead, which resolves by
  shape-verified suffix matching (§27).

## Important environment notes

- Hardware only: board is a physical KV260 at `192.168.137.126`
  (`root`/`<BOARD_PASSWORD>` as of 2026-08-28 -- password has changed at least
  once since this doc was first written; if it's wrong again, ask),
  reachable via Ethernet from the Windows PC running
  Vivado. Board IP must be reset after every reboot:
  `ip addr add 192.168.137.126/24 dev eth0 && ip route add default via 192.168.137.1`
- Vivado project lives on Windows at
  `C:\Xilinx\projects\kv260_custom_noDPU\KV260.xpr` — **this file
  is NOT included in this package** (too large/binary). Vivado steps
  are run as Tcl commands in Vivado's GUI/console.
- The block design has an IMPORTANT NAMING QUIRK: many AXI nets/ports
  are labeled with a stale prefix (e.g. a net physically wired to
  `smartconnect_hp2` may be internally named `smartconnect_hp0_...`)
  because of repeated rewiring across sessions. ALWAYS resolve nets/pins
  by their actual connection (`get_nets -of_objects [get_pins ...]`),
  never trust a name pattern alone.
- All 9 IPs' verified control register addresses, the full Vivado build
  history, every Tcl command that worked, and the complete JTAG/ILA
  setup procedure are documented in detail in `PROJECT_HISTORY.md`.
  Reference it constantly rather than re-deriving things.

## Session startup checklist (board is powered off between sessions)

Everything persists on disk; only the loaded bitstream and the IP address
are lost on power-off. To get back to a working state:

1. Power on the board, then on the board's console re-apply networking
   (does NOT survive reboot):
   ```
   ip addr add 192.168.137.126/24 dev eth0
   ip route add default via 192.168.137.1
   ```
2. From the PC, confirm reachability (SSH password is `<BOARD_PASSWORD>`):
   ```powershell
   & "C:\Program Files\PuTTY\plink.exe" -ssh -batch -pw "<BOARD_PASSWORD>" `
       root@192.168.137.126 "uptime"
   ```
   (first connect needs `-hostkey SHA256:YNosFSx1Q/13iPt+28J9WWUiLAvYBvv3UhA6i7vifa0`)
3. Load the custom design — **via xmutil, never JTAG**:
   ```
   xmutil unloadapp ; xmutil loadapp kv260-custom
   ```
4. Re-verify the hardware still works:
   ```
   python3 validate_all_9_ips.py      # expect 9/9
   ```

Board-resident files that persist: the fixed `hw_ip_driver.py`,
`ip_wrappers.py`, all test scripts, `weights/`, `lcam_v5.xmodel`,
`WEB09971.jpg`. If `hw_ip_driver.py` on the board ever looks stale, re-copy
it from `board_driver/` — it must contain `MIN_CMA_ALLOC` (§26).

## Folder contents

- `PROJECT_HISTORY.md` — full chronological project log (READ THIS FIRST)
- `hls_ip_sources/` — all 9 IPs' HLS C++ source (`.cpp`/`.h`/testbench).
  `head_transpose/head_transpose.cpp` is the CORRECTED version (single
  unified control interface) from the last debugging round — the other
  8 IPs still use the OLD two-interface (`control`+`control_r`) pattern
  and have NOT been updated with this fix (since it was ruled out as
  the root cause, there was no reason to propagate it further).
- `board_driver/` — Python driver for talking to the IPs from the board
  (`hw_ip_driver.py` ← **contains the §26 fix**, `ip_wrappers.py`) plus
  diagnostic scripts from earlier debugging rounds. Also:
  - `weight_resolver.py` — shape-verified weight lookup; use this INSTEAD
    of the broken `weight_map.json` (§27)
  - `extract_weights_fixed.py` — correct xir extraction API. Runs clean
    but returns empty data on a *compiled* xmodel (weights live in a DDR
    blob the Python binding won't expose). Kept as documentation of that
    finding; would work on an uncompiled/float xmodel.
- `debug_scripts/` — self-contained diagnostic + benchmark scripts.
  The ones that matter now:
  - `validate_all_9_ips.py` — **the main demo**: runs all 9 IPs, prints
    PASS per IP
  - `bench_lcam_vs_cpu.py` — LCAM IP vs CPU at real sizes (the 2.05x result)
  - `bench_conv2d_throughput.py` / `estimate_full_network_time.py` — the
    §28 performance measurements
  - `where_is_buf.py` — the script that found the root cause (proves CPU
    buffers live in high DDR, unreachable by the PL)
  - `alloc_size_vs_address_probe.py` — shows the 64KB CMA threshold
  - `test_head_transpose_lowmem.py` — first end-to-end correct HW result
  - older: coherent-port test, post-pragma-fix test, pinpoint tracer
- `notes/` — xclbin metadata JSON templates, original planning doc,
  original inference script, deployment shell script

## Additional files (added after initial packaging)

- `vitis_hybrid/` — **Option B (§31), added 2026-08-30.** The Vitis
  `v++ --link` flow that finally produces DPU + LCAM in ONE xclbin with
  timing met. Contains the extensible-platform Tcl (`build_platform.tcl`,
  `make_xpfm.tcl` — the missing piece that made every earlier `v++`
  attempt fail), the link config (`hybrid.cfg`), the kernel-flow HLS
  script + source (`run_xo.tcl`,
  `lcam_attention_gate_opt_kernel.cpp` — a SEPARATE copy; the IP-flow
  source is untouched), and `package/` for the board. Read its
  `README.md` first. **Built but not yet hardware-verified.**
- `notes/custom-devicetree.dtsi` — the hand-authored device tree overlay
  (HSI-generated content, manually rewrapped as an overlay fragment, plus
  the critical hand-added `zyxclmm_drm`/`xlnx,zocl` node without which no
  DRM device is created and the whole driver path fails). This is real
  project source, not a tool byproduct.
- `notes/EXPLAINER_for_professor.md` — plain-language writeup of the whole
  approach with definitions (DPU, HLS, AXI, XSA, device tree, xclbin,
  zocl, xmutil) — useful orientation if you want the concepts explained
  without the debugging detail.
- `vivado_scripts/` — RECONSTRUCTED Vivado Tcl scripts (synthesis,
  implementation, JTAG/ILA) built from the exact commands documented as
  working in `PROJECT_HISTORY.md`. See `vivado_scripts/README.md` for
  six hard-won gotchas that each cost real debugging time to discover.

## Known NOT included (intentionally, all confirmed non-blocking)

- `KV260.xpr` and the Vivado block design — Windows-side, large/binary
- `decode.cpp` / `nms_top` sources — separate DPU-era Vivado project
- `zynq_ioctl.h` — external Xilinx GitHub source, cited by URL in history
- Tool-generated byproducts (`xhead_transpose_hw.h`, `pl.dtsi`,
  `*.dtbo`, `*.xclbin`, `*.bit`) — all reproducible from what's here
- Board-resident files (`layer_plan.json`, `weight_map.json`, extracted
  weight `.npy` files, `bench_e2e.py`) — live on the KV260 at `~/`
