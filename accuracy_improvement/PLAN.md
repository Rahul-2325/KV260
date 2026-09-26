# Accuracy-Improvement & Audit Workstream — Plan

Triggered by reading Mavros & Katsaros, "SpaSE-UNet3D: Sensor-Driven Wildfire
Detection and Progression Prediction from VIIRS Multispectral Imagery,"
Sensors 26(16):5116 (2026). Different domain (VIIRS satellite time-series
segmentation vs. our UAV image/video object detection) — nothing
architectural to port. What's transferable is methodology: how they turned
a rigorous dataset audit and honest protocol reporting into a citable
contribution alongside their model. This plan adapts that to our own
project, cross-referenced against what we've already measured
(`MY_RESULTS_SUMMARY.md`, `PROJECT_HISTORY.md`).

**Status (2026-09-27): superseded in scope by §0 below.** Sections 1-6 are
kept as-is (still useful methodology and still-relevant findings), but the
project's direction has moved from "improve LCAM-YOLOX's training" to
"find a better architecture and re-apply the same hardware methodology to
it." Read §0 first.

---

## 0. STRATEGIC REDIRECT (2026-09-27)

> **See `RESEARCH_DIRECTIONS.md`** (same folder) for the full cross-layer
> survey — model, training, quantization, DPU config, runtime, PL hardware —
> ranked against the measured frame-time breakdown, plus the key competing
> paper (Karki et al. 2026, attention *approximation* vs our *exact*
> accelerator). It supersedes §0.2b's "width 0.25 first" call: channel
> pruning of v7 (C1) and a pipelined real-video loop (A1) rank higher.

**Decision**: stop iterating on LCAM-YOLOX's training recipe alone. Instead,
use the literature (`LITERATURE_REVIEW.md`) and this project's own findings
to identify a genuinely better-performing architecture (accuracy AND/OR
FPS), then re-apply the proven hardware methodology — the `v++ --link`
DPU+custom-HLS-accelerator hybrid flow, the whole deployment pipeline, the
kernel-protocol lessons (§2.3 of `MY_RESULTS_SUMMARY.md`) — to whatever
DPU-incompatible operation the new architecture has. The hard-won hardware
engineering is reusable; only the model and the specific accelerated op
would change.

**What stays, what's in question**:
- STAYS, unconditionally: the `v++ --link` hybrid build flow, the platform/
  kernel-protocol fixes, the deployment/packaging pipeline, the measurement
  methodology (bit-exactness checks, INA260 power measurement, real-video
  throughput protocol, the confusion-matrix audit tooling). None of this is
  LCAM-specific — it all transfers to any DPU + custom-CU design.
- IN QUESTION: the LCAM module itself, and therefore the specific
  `lcam_attention_gate_opt` HLS kernel built for it. If the new architecture
  uses a different bottleneck op, a new HLS kernel gets designed for THAT op
  using the same design lessons (rounding-mode correctness, AXI port-width
  matching the physical HP-port width, `ap_ctrl_chain` handling, FIFO-depth/
  BRAM/timing-slack trade-offs — all documented and reusable).
- The currently-running Kaggle `lcam-v7-retrain` job (v5 LCAM, COCO-pretrained
  init, proper Mosaic/MixUp/EMA/val-selection) is NOT wasted regardless of
  outcome — it's a legitimate, better-trained LCAM-YOLOX baseline to compare
  any new architecture against, and confirms the retraining pipeline itself
  works.

### 0.1 Concrete FPS levers identified (2026-09-27), for either architecture

**Architecture/training-side:**
1. Lighter backbone width (`width=0.25` vs. current `0.5`) — halves backbone
   compute; DPU compute (~41-43 ms/frame) is still the dominant cost in the
   pipelined-throughput ceiling.
2. Simplify the attention module itself — LCAM currently runs
   ChannelAttention THEN SpatialAttention sequentially, each with its own
   gate+multiply (2x the custom-accelerator work per instance). An ablation
   showing SpatialAttention's accuracy contribution is small would justify
   dropping it — halves the accelerator's per-instance cost as a side effect
   of a legitimate architecture simplification, not a hardware hack.
3. Smaller input resolution (512x512 vs 640x640) — direct compute cut,
   accuracy trade-off not yet quantified.

**Hardware-side (identified earlier, never finished — still on the table
for whichever architecture is chosen):**
4. The unsolved read-back bottleneck (`PROJECT_HISTORY.md` §35-37): ~21 ms/
   frame lost to an uncached read out of the custom CU's output buffer — the
   single largest remaining serial cost in the gate call. Two fixes were
   tried and killed (pagemap-based zero-copy: physical address lands in
   HIGH DDR, unreachable by the PL; a `CACHEABLE` zocl flag that turns out
   not to exist). NOT tried: the C++ VART API, where
   `TensorBuffer::data_phy()` actually exists (the Python binding only
   exposes `get_tensor()`). This is a scoped, concrete, unfinished lead.
5. Fold the trailing head ops (sigmoid-LUT, identity-transpose) into the
   SAME custom-CU invocation instead of leaving them as separate CPU glue —
   removes the last ~2.7 ms/frame without needing a second compute unit
   (this device supports the DPU plus roughly one additional custom CU, not
   two).

### 0.2 Candidate architectures to evaluate (from LITERATURE_REVIEW.md §6)

Criteria: (a) reports better accuracy and/or speed than plain YOLOX-based
fire/smoke detectors in the literature, (b) has SOME operation a standard
Vitis-AI DPU compiler is unlikely to map natively (an attention gate, a
non-standard activation, a fusion op) — without that, there's nothing left
for a custom accelerator to do, and the hardware contribution disappears.
Shortlisted, not yet evaluated in depth:
- **YOLO-FireAD** (arXiv 2505.20884) — attention-guided inverted residual
  learning + dual-pooling; the attention-guided part is the interesting bit.
- **EFA-YOLO** (arXiv 2409.12635) — "efficient feature attention," another
  attention module in the fire/flame-detection space, same category as LCAM.
- **ESCFM-YOLO** (MDPI Appl. Sci. 16(2):778) — lightweight dual-stream
  architecture, explicitly targets edge devices already.
- **YOLOFM** (Sci. Reports, YOLOv5n-based) — lighter base detector (v5n vs.
  our YOLOX-s), worth checking if its accuracy/speed trade beats ours before
  even considering a hardware angle.

None of these have been read in depth yet, and none has been checked for
"does it have a DPU-incompatible op worth building hardware for" — that
check is the actual next step, not just picking the highest reported
accuracy number.

### 0.2b Candidate papers actually read (2026-09-27) — filtering result

Extracted and read the architecture sections of all four shortlisted papers
(pypdf text extraction, since `pdftoppm`/PDF rendering isn't available on
this machine — worth fixing later if visual figures matter).

| Paper | Params | Attention mechanism | Base framework | Verdict |
|---|---|---|---|---|
| YOLO-FireAD | 1.45M (vs our 8.98M) | AIR block: input split into Q/K/V by 1x1 conv, each of Q/K run through a spatial gate (`x⊙σ(Conv3x3(DWConv(x)))`) then a channel gate (`x⊙σ(MLP(GAP(x)))`), fused as `DWConv3x3((Q̂+K̂)⊙V)` — two sigmoid gates and a 3-branch elementwise fusion, genuinely MORE complex than LCAM's single gate | YOLOv8n (Ultralytics) | Equations are precisely given (reproducible), but a materially bigger/riskier HLS kernel than LCAM was, with no evidence yet that it's actually DPU-incompatible the way LCAM's gate is |
| EFA-YOLO | 1.4M | EAConv/EADown — paper never gives the actual gate formula, only prose description | unspecified | Not safely reproducible from the paper alone; would need their GitHub repo. Deprioritized. |
| ESCFM-YOLO | 1.89M | Two SEPARATE detector heads/pipelines (fire-only, smoke-only), not a joint 2-class detector | YOLOv5n | Real architectural redesign (dual-stream), not a drop-in attention swap. Bigger lift than the others. |
| YOLOFM | — | Grab-bag: QARepVGG re-parameterized conv + a new decoupled head (NADH) + a new loss (Focal-SIoU) — no single clean attention module to target for hardware | YOLOv5 | No one clear DPU-incompatible bottleneck to build a kernel for. Deprioritized. |

**Decision**: don't port YOLO-FireAD's AIR attention wholesale — it's a
bigger, unproven-for-DPU redesign of the exact kernel-design risk this
project already spent real engineering time de-risking for LCAM. Instead,
take the one lesson that generalizes cheaply and safely: **all four
candidates get competitive accuracy at 1.4-1.9M params, vs. our 8.98M**.
The DPU compute cost (~41-43 ms/frame) is still the dominant term in the
current pipelined-throughput ceiling (§0.1 item 1), so a lighter backbone
attacks the actual bottleneck directly, while KEEPING the proven LCAM gate
and its already-working, hardware-verified HLS kernel — a real, different
architecture (not just "LCAM-YOLOX retrained again"), at much lower
engineering risk than adopting an unproven new attention mechanism with
no time left to debug a failed HLS synthesis.

The more ambitious option (design a custom kernel for YOLO-FireAD's AIR
block) stays documented here as a stretch goal if time allows later, not
abandoned — just not the next thing to build.

### 0.3 Next steps

1. Wait for `lcam-v7-retrain` to finish — gives a real, better-trained LCAM-
   YOLOX number to use as the baseline the new architecture needs to beat.
2. Read the 4 shortlisted papers' architecture sections specifically for:
   what exact operation their attention/efficiency module performs (to
   assess DPU-compilability the same way §2.3 of `MY_RESULTS_SUMMARY.md`
   worked out for LCAM's Hardsigmoid-vs-gate-multiply split), and their
   reported accuracy/speed on whatever dataset they use (not directly
   comparable to ours, but informative).
3. Pick one (or rule all four out and look further) based on: real
   DPU-incompatible bottleneck present, and plausible accuracy/speed
   improvement over the v7-retrained LCAM-YOLOX baseline.
4. Only then: build the training pipeline for it (reusing the proven
   git-clone-YOLOX + patch-backbone + proper-Exp-config Kaggle approach from
   §6 below, not the hand-rolled fine-tune scripts), and only after THAT
   design the new HLS kernel and hybrid bitstream.

---

## 1. Formalize the label-quality audit (extends §8.3 of MY_RESULTS_SUMMARY.md)

What we have: two specific mislabeled images found by inspection
(`WEB03809`, `PublicDataset01011` — blazing fire labeled "smoke" in ground
truth), and the observation that filenames carry distinct source prefixes
(`AoF*`, `WEB*`, `PublicDataset*`) implying inconsistent original annotation
efforts. This is one paragraph today.

What SpaSE-UNet3D §4 does with the equivalent problem: systematically audits
every item in every split, publishes exclusion/inconsistency counts as a
table, and treats the audit as a contribution in its own right (their
abstract leads with it).

Proposed for our paper:
- Group the 3,099-image labelled set by filename source prefix.
- For each source, run the `--confusion` class-agnostic matching
  (already built, `eval_map.py`) and report the fire/smoke confusion rate
  PER SOURCE, not just the aggregate 0.6%. If one source (e.g.
  `PublicDataset*`) is disproportionately responsible for the confusion
  cases, that is itself a finding, and mirrors their observation that BA
  labels have different quality than AF labels.
- Publish the full list of confusion cases (not just the two we manually
  inspected) as a table/appendix, the way they publish exclusion lists.
- State plainly, as they do, that this is a property of the dataset, not
  of the hardware or the model.

## 2. Multi-seed retraining for variance reporting

Current state: mAP@0.5 = 0.7360, mAP@[.5:.95] = 0.3774 — a single trained
model, evaluated once (hardware vs. software control both draw on this
same trained model, so the 4-decimal-place agreement in §38.1/§5 of
MY_RESULTS_SUMMARY.md is a HARDWARE-vs-SOFTWARE correctness result, not a
statement about training variance).

Proposed: retrain the float model N times (SpaSE-UNet3D uses 5 seeds) with
identical hyperparameters, differing only in random seed, then quantize
and evaluate each the same way. Report mAP as mean ± std. This does three
things for the paper:
- Makes the accuracy claim defensible the way theirs is.
- Gives an honest denominator for "is 0.7360 a typical result or a lucky
  seed" — currently unanswered.
- If std is small (their AF task: σ=0.0005-0.0009), that is itself worth
  stating as a stability result, the same way they do.

## 3. Protocol-sensitivity table (extends the existing --confusion / threshold tooling)

They show scoring convention alone moves F1 by 0.10 — comparable to the
spread between published baselines — and document it explicitly (Table 3/
Table 4) so a reader knows what a reported number actually means.

We have the pieces (`eval_map.py --conf`, `--nms`, `--confusion`) but have
only ever reported ONE operating point (conf=0.30) in the writeup. Proposed:
a small table sweeping confidence threshold (e.g. 0.20-0.50) and reporting
mAP, detection count, and confusion rate at each, so a reader can see how
sensitive our headline number is to the chosen threshold — mirroring their
Table 4 decomposition.

## 4. Statistically grounded ablation of the LCAM attention module (accuracy side)

We already have a HARDWARE-cost ablation of the LCAM kernel (§39.9:
bit-width narrowing and DSP-bind pragma changes, both backfired, LUT-level).
We do NOT have an ACCURACY-side ablation showing the LCAM attention gate is
worth including at all, in the way their Table 8 isolates SE attention's
contribution (+0.0005 F1 without it, i.e., statistically indistinguishable
at their sample size — and they report that honestly rather than
overclaiming).

Proposed: train the detector with the LCAM gate present vs. replaced by
identity (or a plain 1x1 conv) at a fixed epoch budget, multiple seeds each,
same protocol as item 2. Report Δ and Δ/σ the way they do. Possible
outcomes, both publishable:
- LCAM measurably improves accuracy beyond seed noise → strengthens the
  paper's motivation for building custom hardware for it.
- LCAM's accuracy contribution is within noise, same as their SE finding →
  reframes the hardware contribution honestly as "this op is expensive on
  the DPU's software fallback regardless of its accuracy contribution,
  and we accelerate it" rather than overclaiming an accuracy benefit — a
  defensible, SpaSE-UNet3D-precedented way to report a null accuracy
  result without it undermining the hardware contribution.

## 5. Explicitly not doing

- Not porting SpaSE-UNet3D's architecture (3D U-Net, ASPP, temporal SE
  pooling) — wrong task (segmentation, not detection), wrong input
  (multispectral satellite time-series, not RGB UAV images/video). No
  architectural relationship beyond "both use squeeze-and-excitation-style
  channel attention," which is already what LCAM is.
- Not re-litigating the 9-custom-IP hardware work (per standing
  instruction, excluded from `MY_RESULTS_SUMMARY.md`) — this workstream is
  entirely about the training/accuracy/dataset side, not the hardware.

---

## 6. Training-approach audit (found 2026-09-26, reading the actual training code)

The model was trained via `~/wildfire_project/kaggle_upload/finetune_lcam_v3.py`
(WSL, on this PC) on a Kaggle notebook (free T4 GPU) — confirmed necessary
since the local Vitis-AI Docker (`xilinx/vitis-ai-pytorch-cpu:...`) is
CPU-only. Architecture: YOLOX-s (`depth=0.33, width=0.5, act=lrelu`,
2 classes) + the LCAM module from `yolox/models/lcam.py` (NOT `lcam_v5.py`,
which despite the name is a same-sized, unused leftover file — the real
model uses whichever module `darknet.py` imports, which is `lcam.py`).
Fine-tuned from an earlier checkpoint, 20 epochs, batch 16, SGD lr=1e-4,
cosine schedule.

Reading the training loop end to end surfaced four gaps, independent of
anything from the SpaSE-UNet3D paper, and arguably a bigger lever on mAP
than the paper-inspired items above:

- **No validation pass during training.** `val_loader` is constructed but
  never iterated in the loop; "best" checkpoint is selected purely by
  lowest TRAINING loss. No validation-mAP-based model selection at all.
- **Minimal augmentation**: only a random horizontal flip. No Mosaic,
  MixUp, or multi-scale training — all standard in YOLOX's own official
  recipe (this is a hand-rolled loop, not YOLOX's built-in `Trainer`).
- **No EMA weight averaging** — YOLOX's official recipe uses this for the
  reported checkpoint; absent here.
- **Single run** — same "no variance reported" gap as item 2 above, now
  with a concrete script to add seeding to.

Also confirmed (resolves the earlier hard-sigmoid open question, §11 of
MY_RESULTS_SUMMARY.md if that section gets written): dumping the compiled
`lcam_hybrid.xmodel`'s CPU subgraph op types directly (via the Vitis-AI
Docker's `xir` bindings) shows the four LCAM gate subgraphs contain only
`fix2float / float2fix / mul` — Hardsigmoid genuinely runs on the DPU
natively, exactly as `lcam.py` intends. The "sigmoid" CPU subgraphs
found in §34/§36 belong to the YOLOX detection head, not LCAM, and are
architecturally unrelated. No contradiction with the training code.

**Proposed retraining plan, revised**: rather than a from-scratch
overhaul, re-run `finetune_lcam_v3.py` with (a) an actual validation pass
each epoch selecting best-by-mAP not best-by-loss, (b) YOLOX's standard
Mosaic/MixUp augmentation restored, (c) EMA, (d) 3-5 seeds — then apply
items 1-4 above (source-grouped audit, protocol-sensitivity table,
LCAM ablation) to the resulting checkpoint(s). This should be done as a
modified copy of `finetune_lcam_v3.py`, keeping the original untouched
for reproducibility of the currently-deployed v5 model.

## Open questions before any of this can actually run

This plan is retraining work, which needs infrastructure this session
hasn't established access to yet:

1. **Where does training happen?** The original LCAM-YOLOX float model was
   presumably trained before this hardware project began (PROJECT_HISTORY.md
   references a WSL Vitis-AI Docker workspace for quantization/compilation,
   not training). Is the training codebase (dataset loader, model
   definition, training loop) available on this Windows machine, in WSL, or
   somewhere else entirely (a separate GPU machine, a cloud notebook)?
2. **Is there a GPU available for retraining runs**, and if so where —
   this Windows PC, the WSL environment, or elsewhere?
3. **Which item above should be tackled first?** Items 1 and 3 (the label
   audit and the protocol-sensitivity table) need no retraining at all —
   they run entirely on the existing 400/3,099-image labelled set and the
   already-deployed `eval_map.py`, so they could start immediately, on the
   board or even off it. Items 2 and 4 need an actual training pipeline
   and multiple training runs, which is real GPU time.
