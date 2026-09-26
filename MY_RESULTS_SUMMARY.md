# MY_RESULTS_SUMMARY.md — LCAM-YOLOX Hybrid DPU+HLS Accelerator on KV260

Clean, paper-ready summary of this project's own measured results, compiled
for NotebookLM ingestion alongside `LITERATURE_REVIEW.md`. Every number here
is hardware-measured (not simulated/estimated) unless explicitly marked
"projected." Source of truth for anything not fully reproduced here is
`PROJECT_HISTORY.md` (chronological engineering log — this file extracts
only what belongs in a paper).

**Scope note, deliberate**: this file covers the DPU + single custom LCAM
accelerator hybrid design only — the design that actually works, is
hardware-verified, and is deployed. An earlier exploratory phase attempted
converting all 9 originally-planned custom IPs into additional compute
units (multi-CU builds, up to 10 CUs alongside the DPU). That work is
excluded here by design — it did not fit/close timing on this device and is
not part of this project's reported contribution.

---

## 1. Problem statement

Xilinx's DPU accelerates standard CNN operators (conv, pool, eltwise) but
cannot natively execute the LCAM (Layer-wise/Local Channel Attention
Module) attention gate used in LCAM-YOLOX. When the model runs on the DPU
alone, every LCAM gate falls back to software on the ARM Cortex-A53,
which dominates end-to-end latency. This project builds a dedicated HLS
hardware accelerator for the LCAM gate and fuses it with the DPU in one
Vitis-linked bitstream, so the entire model — DPU convolutions plus the
attention mechanism — runs in hardware with no CPU-side attention
computation.

Baseline measured on this hardware, DPU-only with CPU-side LCAM fallback:
**426.51 ms/frame, 2.34 FPS** (10 runs, mean; median 426.20, std 0.98 ms).
DPU compute alone (8 subgraphs): 41.15 ms. CPU fallback: 385.36 ms (90.4%
of the frame), of which the four LCAM gates measured 200.30 ms
(`DEEPHI_PROFILING=1`, per-op microsecond timestamps, averaged over 9
frames) and everything else on the CPU (head transpose/reshape/concat/
sigmoid ops) measured ~193 ms.

---

## 2. How the custom LCAM IP was built

### 2.1 Original IP
First working version: HLS C++ kernel, 8-bit AXI4 memory port (1 int8 per
beat), floating-point multiply + `roundf()` for the requantization step,
synthesized with an `fmul_32ns_32ns_32` DSP multiplier IP. Functionally
correct but slow: 40.60 ms for all four gate layers on real hardware
(standalone bitstream, 100 MHz).

### 2.2 Optimized IP — the version actually deployed
The entire gate operation reduces to integer arithmetic with no float
round-trip needed:
```
out = clamp( sign(p) * ((|p| + 2^(s-1)) >> s) ),  p = feat * weight,
s = fp_feat + fp_weight - fp_out
```
Changes made:
- AXI memory port widened 8-bit → 512-bit (64 int8 values/beat)
- Float multiply + `roundf()` replaced by a pure integer shift (no DSP
  float unit; final design uses 5 DSPs for the multiply itself)
- Inner loop unrolled 64 lanes wide, `PIPELINE II=1`

**A rounding correctness subtlety that would have silently corrupted
~2% of outputs**: `roundf()` rounds ties away from zero, but a bare
`(p + bias) >> s` rounds ties toward −∞. On signed int8 data (negatives are
common) these disagree on exact half-integer cases — e.g. `roundf(-2.5) =
-3` vs. naive shift `= -2`. Fix: round the *magnitude* and reapply the
sign. Verified bit-identical to the reference over the **full int8 × int8
domain** for all four layer configurations (`verify_lcam_int_math.py`);
the naive version would have gotten 896–1,296 cases wrong per
configuration.

HLS synthesis result (Vitis HLS 2022.2, `xck26-sfvc784-2LV-c`, 100 MHz
target): estimated timing 7.3 ns (10 ns target), DSP 5, FF 7,194,
LUT 40,296 (before the later 512→128-bit narrowing described in §2.3),
BRAM 0, URAM 0 — no floating-point units at all in the final design.

Standalone measurement (isolated bitstream, before fusing with the DPU),
four real network layer shapes:
```
layer  shape          HW (ms)   CPU (ms, numpy exact-rounding)   speedup
23     160x160x64        2.24      76.35                          34.0x
41     80x80x128         0.61      36.10                          58.9x
53     40x40x256         0.33      17.98                          54.9x
83     20x20x512         0.31       7.54                          24.1x
--------------------------------------------------------------------------
TOTAL                    3.50     137.97                          39.5x
```
11.6x faster than the original (float) IP; correctness bit-exact on every
layer.

### 2.3 Fusing with the DPU — the `v++ --link` kernel flow
Two integration routes were tried. The first (Vivado IP-integrator block
design) produced a bitstream, but XRT could not drive the vendor DPU
correctly inside it — the Vivado-TRD DPU is a raw IP with no `kernel.xml`,
so XRT drove it with the generic `ap_ctrl_hs` handshake and hung the
compute unit by writing `ap_start` into what is actually a read-only DPU
version register. This is a protocol mismatch, not a wiring bug, and no
amount of xclbin metadata editing fixed it.

The working route: package **both** the DPU and the LCAM kernel through
Vitis's `v++ --link` kernel flow, on a custom-built extensible hardware
platform (PS + clocking + AXI interconnect + PFM metadata, since the
platform originally in use published no clocks/AXI ports and could never
have accepted linked kernels). This gives the DPU its own proper
`kernel.xml` so XRT/VART drive it with the correct protocol.

Two source-level changes were required to make the LCAM kernel a legal
Vitis kernel (as opposed to a legal Vivado IP, which it already was):
- All AXI-Lite control ports must share one bundle name (`bundle=control`
  for every scalar/offset argument) — Vitis kernel mode hard-errors on the
  two-bundle pattern that was harmless in the IP-integrator flow.
- The memory port width matters differently than in the IP flow: a ZynqMP
  `S_AXI_HP` port is physically 128 bits wide, so a 512-bit kernel port
  forces the platform to insert a width converter and the extra unrolled
  lanes sit idle 3 of every 4 cycles. Narrowing the port to 128 bits (16
  lanes instead of 64) cost **zero** measured throughput — the AXI bus, not
  the arithmetic, was always the limit — while roughly halving placed LUT
  usage (28,993 → 10,062 placed LUT for the kernel).

**A second, separate control-protocol trap**: kernels built through
`v++`/Vitis HLS's `-flow_target vitis` use `ap_ctrl_chain`, not the
`ap_ctrl_hs` protocol the original standalone IP used. Under
`ap_ctrl_chain`, `ap_done` is sticky and must be explicitly cleared by
writing `ap_continue` — a driver written for `ap_ctrl_hs` (which
self-clears `ap_done` on read) will silently "complete" every call after
the first in ~0.03 ms against a stale, untouched output buffer, with no
error. Fixed by writing the correct sequence:
`wait ap_idle → write args → ap_start → poll ap_done → write ap_continue`.

**A DPU-specific integration bug found and fixed**: after fingerprint
matching succeeded, DPU calls still timed out. Root cause: `v++` cascades
every kernel's interrupt through the platform's shared `axi_intc_0` into
one PS interrupt line, while `zocl` (the Linux DRM/KDS driver) registers
one direct GIC interrupt line *per compute unit* — nothing programs the
`axi_intc`'s enable registers, so no interrupt ever propagates, even
though the DPU itself computes correctly (confirmed by watching its
`ap_ctrl` register directly). Fix, no rebuild required: run XRT in polling
mode (`ert_polling=true` in `xrt.ini`), which makes KDS poll the CU status
register instead of waiting on the missing interrupt.

---

## 3. Hardware resource utilization — final deployed design

Two build variants exist; **`kv260-hybrid3` is the recommended/current
one** (better slice headroom at negligible throughput cost — see §3.2).
Both are hardware-verified and independently loadable on the board.

### 3.1 `kv260-hybrid2` — first hardware-verified hybrid build
```
WNS +0.031 ns   TNS 0.000   0 failing endpoints (of 362,093)

CLB LUTs        61,903 / 117,120   52.85%
CLB Registers  110,807 / 234,240   47.30%
CLB slices      14,201 /  14,640   97.00%   <- tightest constraint
Block RAM Tile     98.5 /    144   68.40%
URAM                 46 /     64   71.88%
DSP48E2              716 /  1,248   57.37%

  DPUCZDX8G_1                48,996 LUT   82 BRAM   46 URAM   710 DSP
  lcam_attention_gate_opt_1  10,062 LUT    7 BRAM    0 URAM     6 DSP
```
Compute units, from `xclbinutil --info`: `DPUCZDX8G_1 @ 0xa0010000`,
`lcam_attention_gate_opt_1 @ 0xa0020000`.

### 3.2 `kv260-hybrid3` — BRAM/slice-optimized rebuild (recommended)
Two changes from hybrid2, both DPU/kernel-config only — no RTL logic
changed:
- DPU: `dpu_conf.vh` URAM bank counts raised (`def_UBANK_IMG_N` 5→7,
  `def_UBANK_WGT_N` 17→21). Trades BRAM for URAM: 82→67 BRAM for +4 URAM.
- LCAM kernel: AXI adapter FIFO depth reduced (`num_outstanding=16,
  max_burst=64` → `4, 32`). This block RAM was never datapath storage
  (csynth reports 0 BRAM_18K for the kernel itself) — it was purely the
  `m_axi` burst buffers, sized `outstanding × burst`.

```
                       hybrid2 (before)         hybrid3 (after)          delta
CLB LUTs            61,903 / 117,120 52.85%   61,554 / 117,120 52.56%
CLB Registers      110,807 / 234,240 47.30%  111,010 / 234,240 47.39%
CLB slices          14,201 /  14,640 97.00%   14,018 /  14,640 95.75%   more headroom
Block RAM Tile         98.5 /   144  68.40%       81 /    144  56.25%   -17.5 tiles (-17.8%)
URAM                     46 /    64  71.88%       50 /     64  78.13%
DSP48E2                 716 /  1,248 57.37%      716 /  1,248  57.37%   unchanged

  DPUCZDX8G_1               82 BRAM  46 URAM  48,996 LUT   67 BRAM  50 URAM  48,932 LUT
  lcam_attention_gate_opt_1  7 BRAM   0 URAM  10,062 LUT    4 BRAM   0 URAM   9,771 LUT

WNS   +0.031 ns  ->  +0.018 ns     both MET, 0 failing endpoints
```
DPU fingerprint **unchanged** (`0x101000012010407`) — the same compiled
xmodel (`lcam_hybrid.xmodel`) loads on both builds with no recompile.

**Hardware-verified trade-off** (both builds tested on the physical
board): single-frame latency and CU execution time are *identical*
between the two builds (2.18 ms CU time, ~75.6–75.9 ms single-frame
latency either way — the FIFO reduction costs nothing when frames don't
overlap). Under concurrent/pipelined load, hybrid3 costs a small,
reproducible throughput regression:
```
                        hybrid2   hybrid3    delta
pipelined, 2 workers   18.73 FPS  18.02 FPS   -3.8%
pipelined, 4 workers   19.43 FPS  19.28 FPS   -0.8%
mAP@0.5 / mAP@.5:.95   0.7360 / 0.3774   identical, both builds
```
Net trade measured: **-17.5 BRAM tiles (-17.8%) and -1.25 percentage
points of slice occupancy, for -0.71 FPS under 2-worker pipelining**
(reproduced across 3 repeat runs: 18.02, 18.02, 18.01 FPS — real signal,
not noise). A further FIFO increase to `8/32` (attempting to recover the
lost throughput while keeping most of the BRAM saving) was tried and
**failed timing closure** (WNS −0.024 ns, 268 failing endpoints) — the
three variables (BRAM, FIFO depth/throughput-under-contention, and timing
slack) trade against each other and cannot all be simultaneously
maximized on this device; hybrid3 is the point on that trade-off actually
shipped.

### 3.3 Why the LCAM kernel costs ~9,771–10,062 LUT for one instance
(Relevant for reviewers who ask why a single attention-gate accelerator
is this large.) Breakdown of the deployed 128-bit/4×32-FIFO kernel:
the dominant cost (~58%) is the main compute loop itself — specifically
**16 unrolled lanes, each needing a full runtime-variable barrel shifter**
(the shift amount `s` is a runtime value, not a compile-time constant,
since it depends on the two operands' quantization scales). Two plausible
micro-optimizations were tried and **both made LUT usage worse**, via a
controlled ablation:
- Narrowing arithmetic to `ap_int<20>` (from the tool-inferred default
  width): LUT went from 14,985 → 21,584 — it defeated HLS's own automatic
  bit-width inference and forced extra width-conversion logic.
- `#pragma HLS BIND_OP ... impl=dsp` (forcing the multiply onto a DSP
  slice explicitly): 14,985 → 15,883 alone; combined with the `ap_int<20>`
  change, 20,944 — the worst of all four configurations tested.
Conclusion, kept: the un-"optimized" default HLS inference was already
the best of the options tried; both interventions were reverted.

---

## 4. Correctness

Three independent levels of bit-exactness were established, from
synthetic to full-dataset:

1. **Arithmetic correctness, full domain**: the optimized integer kernel
   verified bit-identical to the float reference over the *entire*
   int8×int8 input domain, all four layer configurations
   (`verify_lcam_int_math.py`).
2. **End-to-end, single tensor**: running the full model pipeline with
   `--gates ip` vs. `--gates numpy` and diffing the final `[1,8400,7]`
   detection tensor: `max|diff| = 0, mean = 0, 0 of 58,800 elements
   differ`.
3. **End-to-end, full labelled dataset**: scoring the same 400-image
   validation slice (below) with hardware gates vs. software gates:
   **identical mAP to four decimal places, identical 664 detection
   count**, both runs (§5). Confirmed again on 400 real video frames
   (§7): **454 detections, 196/400 frames flagged, byte-for-byte
   identical between hardware and software code paths.**

Moving the attention computation to custom silicon changes **no
detection, no confidence score, no bounding box** — correctness holds
not just on one demo image but across the full diversity of the
evaluation set and a full video sequence.

---

## 5. Accuracy (mAP)

Validation set: 400-image deterministic slice (every 7th image of a
sorted 3,099-image labelled set — `dataset/extracted/data/val`, YOLO-format
labels, 2 classes: fire=0, smoke=1), 208 fire + 292 smoke ground-truth
objects. All-point IoU-interpolated AP.

```
                        gates = custom IP       gates = software (control)
class     objects   AP@0.5   AP@.5:.95        AP@0.5   AP@.5:.95
fire          208   0.7378      0.4103        0.7378      0.4103
smoke         292   0.7342      0.3446        0.7342      0.3446
------------------------------------------------------------------------
mAP                 0.7360      0.3774        0.7360      0.3774
detections            664                       664
latency            75.84 ms                   289.87 ms
```
Identical to four decimal places — the accelerator is free (no accuracy
cost) and gives a **3.82x** latency win within the exact same evaluation
pipeline. Latency was flat across all 400 frames (75.8 ms at every
50-frame checkpoint), so this is not a lucky single sample.
Caveat: 400-image slice of a 3,099-image set, drawn from the val split
(no separate held-out test set was scored).

---

## 6. Latency and throughput

Three genuinely different numbers exist and must not be conflated in the
write-up — each answers a different question about the pipeline's scope.

### 6.1 Hardware-only, single frame (static preloaded tensor, no camera)
```
end-to-end latency, single frame        75.84 ms    13.19 FPS   5.62x vs. baseline
```
### 6.2 Hardware-only, pipelined (Python worker threads overlapping DPU
and non-DPU work; still a static preloaded tensor)
```
1 worker    76.29 ms/frame   13.11 FPS
2 workers   53.39 ms/frame   18.73 FPS   <- sensible operating point
3 workers   51.93 ms/frame   19.26 FPS
4 workers   51.48 ms/frame   19.43 FPS   (reproduced later: 19.33 FPS, §6 repeat run)
```
Feasibility was confirmed before building this (`probe_gil.py`): the
non-DPU numpy/copy work releases Python's GIL (1.72x speedup at 2
threads) and can overlap the DPU's own execution; the DPU itself only
gains 1.22x from 2 threads, correctly, since it is one physical unit that
cannot execute two subgraphs concurrently — the 1.22x is submission
overhead overlap only. **Latency does NOT improve under pipelining — it
gets worse** (75.8 ms → 106.6 ms per-frame latency at 2 workers) even as
throughput improves; this is a throughput/latency trade, and the paper
must state which one is being claimed for any given number.

### 6.3 Real video, camera-to-boxes (the honest deployment-scope number)
Built and measured on 400 real frames of genuine fire/smoke video content
(construction described in §7):
```
                          decode  pre    DPU    gate    CPU   post   TOTAL      FPS
hardware (custom IP) gates  1.8   25.8   43.0   27.4     2.7   7.3  112.03 ms   8.93
software (numpy) gates      2.0   25.9   43.2    0.0   245.6   7.3  326.60 ms   3.06
```
Speed-up on real video: **2.92x** — lower than the 5.6x compute-only
figure, and this gap is itself a finding: none of the compute-only
figures above (§6.1, §6.2) include real per-frame video decode, letterbox
preprocessing, int8 quantization, or postprocessing (NMS, drawing,
encoding) — a camera-fed deployment pays all of these. **State the scope
explicitly every time a throughput number is quoted**: "hardware compute
only" (13.2–19.4 FPS) vs. "camera-to-boxes" (8.9 FPS) are both correct
and answer different questions.

### 6.4 Postprocessing cost, measured separately (never folded into any FPS number above)
```
single-frame path:  decode 15.48 + nms/draw 26.26 + imwrite 25.74 = 67.53 ms
4-worker path:      decode 15.20 + nms/draw 24.44 + imwrite 25.01 = 64.70 ms
```
Comparable in magnitude to the hardware frame time itself — not
negligible, and reported rather than hidden. Streaming video encode
(MJPEG `VideoWriter`, §7) is far cheaper per frame (2.10 ms) than
one-off single-image `cv2.imwrite()` (25.74 ms), since it avoids
per-call file-open overhead.

---

## 7. Power and energy efficiency

Measured via the KV260 SOM's onboard INA260 current/voltage sensor on the
5V input rail (`/sys/class/hwmon`, `ina260_u14`) — a real electrical
measurement of the whole module (PS+PL+DDR), not a vectorless EDA
estimate.
```
MEASURED (INA260)
  idle, design loaded        4.809 W
  running the pipeline       6.360 W mean, 10.110 W peak
  delta (the workload)       1.551 W

Vivado report_power (routed checkpoint, vectorless, for comparison only)
  Total On-Chip               6.896 W   <- exceeds the measured WHOLE-MODULE
                                            input power; provably an
                                            overestimate — quote the
                                            measured 6.36 W, not this.
```
Energy per frame (same bitstream, differing only in software path —
apples-to-apples):
```
                      power     latency     energy/frame
stock GraphRunner    5.339 W   427.70 ms      2.283 J
hybrid pipeline      5.794 W    75.97 ms      0.440 J
                     --------------------------------------
                      0.92x       5.63x         5.19x better
```
The hybrid draws 8.5% *more* instantaneous power but is **5.19x more
energy-efficient per inference**, because it finishes far sooner — the
metric that matters for a battery-powered UAV.

---

## 8. Real video demonstration

### 8.1 Why a new video had to be built
No genuine fire/smoke video existed in the project. Two pre-existing
files that looked plausible by filename (`wildfire_720p.avi`,
`wildfire_pos_720p.avi`) were discovered — by direct inspection, not
assumed — to each be a single static surveillance frame looped 120
times (verified via a 12-frame contact sheet spanning each entire clip:
identical clouds, identical parked cars, an identical fixed timestamp
overlay in every sampled frame). Neither showed any fire or smoke. A
plausible filename and a sane-looking mean pixel value are not proof of
video content — worth stating as a methodological note, since it was
caught only by deliberately sampling frames across a whole clip.

### 8.2 Construction
Built (`build_wildfire_video.py`) from the same 400 labelled photographs
used for the mAP result (§5) — real, varied, ground-truthed fire/smoke
content, not synthetic frames. Source images span >90 distinct native
resolutions (240×320 up to 1920×1080); each is letterboxed (grey-padded)
into a fixed 1280×720 canvas — the single most common native resolution
in the dataset (186/400 images) and the model's own training resolution —
minimizing distortion/resampling for most frames. Output: 400-frame MJPEG
AVI, plus a frame-index → source-filename manifest for ground-truth
lookup.

### 8.3 Fire/smoke class-confusion analysis
Prompted by a visible mislabel in the demo video (a distant smoke plume
tagged `fire 0.79`). First ruled out as a hardware/accelerator defect —
already excluded by every bit-exactness result in §4, including on this
exact video. If the label is wrong, the cause is in the model's learned
weights or the training labels, not the custom silicon.

Built a class-agnostic confusion tool (`eval_map.py --confusion`):
matches every detection to the best-IoU ground-truth box in its image
*regardless of predicted class* (ordinary per-class AP cannot reveal
this — a wrong-class detection just looks like one false positive plus
one separate false negative, with no visible link). Run at conf=0.30 to
match the demo video's own threshold:
```
fire  -> fire    152   CORRECT
fire  -> smoke     2   CONFUSED
smoke -> smoke   208   CORRECT
--------------------------------------------------
correctly classified                    : 360
class confused (right box, wrong label) :   2    (0.6% of matched detections)
no matching ground truth at all         : 104    (hallucinations — separate issue)
```
Only 0.6% of detections landing on a real object had the wrong class
label; the dominant source of imperfect mAP is hallucinated detections
(104), a different problem with a different fix (confidence/NMS
tuning), not fire/smoke class confusion.

**Both of the two actual confusion cases were pulled and visually
inspected** (`WEB03809`, conf 0.848, IoU 0.927; `PublicDataset01011`,
conf 0.562, IoU 0.727): both are images of unmistakable, fully-involved,
blazing fire, where every ground-truth box in the label file is annotated
class "smoke" and none is annotated "fire." **The dataset's own
ground-truth labelling calls these "smoke"; the model's "fire" call is
the visually defensible answer.** This is a labelling inconsistency in
a multi-source scraped dataset (filenames carry distinct source prefixes
— `AoF*`, `WEB*`, `PublicDataset*` — suggesting several inconsistent
original annotation efforts, not one consistent one), not a demonstrated
model weakness. For the paper: report the 0.6% confusion rate and these
two labelling-inconsistency examples together — an honest accuracy
discussion includes both the model's real limitations and the dataset's.

No specific source frame was identified for one particular ambiguous
screenshot (a distant plume with a small visible glow, moderate
confidence); no claim is made about that specific case beyond what the
aggregate 0.6% measurement supports.

---

## 9. Reproducibility / methodology details

- Board: Xilinx/AMD Kria KV260 (Zynq UltraScale+ MPSoC,
  `xck26-sfvc784-2LV-c`), physical hardware, not simulation.
- Tools: Vitis HLS 2022.2, Vivado 2022.2, Vitis 2022.2, Vitis-AI 3.0,
  XRT 2.14.
- DPU: `DPUCZDX8G` architecture, B4096 configuration,
  `CHANNEL_AUGMENTATION_DISABLE`, `RAM_USAGE_LOW`, URAM enabled.
  Fingerprint `0x101000012010407` (both hybrid2 and hybrid3 — the two
  BRAM/URAM-bank configurations described in §3 do not change the
  fingerprint, so the same compiled xmodel runs unmodified on both).
  DPU clock 300 MHz.
- Integration flow: `v++ --link` Vitis kernel flow on a custom extensible
  hardware platform — NOT the Vivado IP-integrator block-design flow
  (which built but could not be driven correctly by XRT/VART for the
  DPU; see §2.3).
- Required runtime setting: `XRT_INI_PATH=/home/root/xrt.ini` with
  `[Runtime] ert_polling=true` — without this, every DPU call times out
  (missing-interrupt issue, §2.3).
- Dataset: 3,099-image labelled fire/smoke set, YOLO-format labels,
  2 classes. mAP evaluation (§5) and the video demo (§8) both draw from
  the same 400-image deterministic slice, so results are directly
  comparable across sections.

---

## 10. Known limitations — state these explicitly, do not omit

- The 400-image evaluation slice is drawn from the validation split; no
  separate held-out test set was scored.
- Compute-only FPS figures (13.2 / 18.7 / 19.4 FPS) exclude real camera
  decode, preprocessing, and postprocessing entirely by design (they
  isolate hardware performance) — only the real-video figure (8.9 FPS,
  §6.3) represents true camera-to-boxes throughput. Never quote a
  compute-only figure as if it were the deployment-scope number.
- Postprocessing (~65–68 ms) is real and non-negligible, comparable in
  magnitude to the hardware inference time itself.
- Pipelining trades latency for throughput — per-frame latency nearly
  doubles (75.8 → 106.6 ms) at the 2-worker operating point even as FPS
  rises; a single-shot/low-latency application should not pipeline.
- The BRAM-optimized build (hybrid3) measurably costs throughput under
  concurrent load (−0.7 FPS at 2 workers) despite being identical at the
  single-frame level — resource savings measured in isolation do not
  always transfer to a concurrent workload.
- mAP@0.5 (0.7360) is meaningfully below mAP@[.5:.95] would suggest is
  achievable with a tighter IoU criterion (0.3774) — typical for this
  class of lightweight detector; not specific to the hardware.
- The 0.6% fire/smoke confusion figure is measured on this project's own
  400-image slice with this project's own confidence threshold (0.30);
  it is a methodology-specific number, not a universal property of the
  model.

---

## Points worth adding once available (flagged, not fabricated)

- A resolved comparison against the anchor paper's own reported 78.11%
  mAP and 195 FPS DPU-only figure — not done yet because the evaluation
  protocols (dataset split, confidence threshold, IoU averaging) are not
  confirmed to match; comparing mismatched protocols would be misleading
  in a journal paper. Worth resolving once the anchor paper's full text
  is read closely (see `LITERATURE_REVIEW.md` §0's open action item).
- A true zero-copy DPU→accelerator→DPU data path was investigated
  (avoiding the ~21 ms/frame uncached read-back that is the single
  largest remaining cost inside the gate call) but both routes tried
  (pagemap-based physical-address translation via the Python `vart` API,
  and a `CACHEABLE` zocl allocation flag) were dead ends — documented in
  `PROJECT_HISTORY.md` §35–37 if this is worth one paragraph as
  identified-but-unsolved future work.
