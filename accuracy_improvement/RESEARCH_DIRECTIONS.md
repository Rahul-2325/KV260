# Research Directions — Where Accuracy and FPS Can Actually Come From

Broad survey (2026-09-27) across every layer of the system: model, training,
quantization, DPU configuration, runtime software, and PL hardware. Every
lever below is tied back to THIS project's measured frame-time breakdown,
not generic advice. No training has been launched from this document —
`lcam-v7-retrain` is the only job running.

---

## 1. Start from the measured bottleneck, not the literature

Real-video, camera-to-boxes, hybrid bitstream (`MY_RESULTS_SUMMARY.md` §6.3):

```
decode   pre    DPU    gate   CPU   post  | TOTAL      FPS
  1.8   25.8   43.0   27.4    2.7   7.3  | 112.0 ms   8.93   (sequential)
                       ^-- of which ~21.5 ms is the uncached read-back
```

Three terms dominate: **DPU compute (43.0)**, **gate (27.4, mostly
read-back)**, **preprocess (25.8)**. Any proposal that doesn't move one of
those three doesn't move FPS. `post` (7.3) and `CPU` (2.7) are already small.

Also: this 112 ms loop is **sequential**. The 18.7–19.4 FPS headline
numbers come from a *pipelined* loop (worker threads overlapping stages),
but that pipelining was only ever applied to a static preloaded tensor,
never to the real-video loop.

---

## 2. Positioning — our accuracy is already strong on D-Fire

Published D-Fire mAP@0.5 figures (different protocols; for orientation, not
a ranking):

| Model | D-Fire mAP@0.5 | Source |
|---|---|---|
| YOLOv8n | ~62.5% | uncertainty-aware post-detection paper (baseline 0.625) |
| SemaFire-YOLO (2025) | 64.3% | beats v5n/v8n/v11n/v12n by 0.6–3.8 pts |
| YOLOv8m | 79.04 ± 0.21% | reported on D-Fire |
| **Ours, LCAM-YOLOX-s v5 (deployed)** | **73.6%** | 400-image val slice, board, `eval_map.py` |
| **Ours, earlier v3.1 proper-Exp run** | **76.5%** | full 3,099-image val, YOLOX evaluator |

Nano-class detectors sit ~10 points below us on D-Fire; we are within a
few points of YOLOv8m (a far larger model). **The accuracy story is not
weak** — it just has to be presented against same-dataset, same-class
comparators, with protocol stated per row (SpaSE-UNet3D lesson).

A 96.3% D-Fire mAP50 claim (CP-YOLOv11-MF fusion) also appears in search
results — almost certainly a different split or fusion input; treat with
suspicion until its protocol is read.

---

## 3. The single most important related paper found

**Karki, Ahmed, Jungeblut (HSBI), "No Attention, No Problem: DPU-Aware
Attention Approximation in Modern YOLO on FPGA," arXiv 2607.13106,
July 2026.**

They hit thesolved it the OPPOSITE way:
**approximate** the attention so  same wall we did — attention ops (reshape, transpose, matmul,
softmax) aren't DPU-native — and it compiles onto the DPU (q⊙k elementwise
instead of matmul, **hard-sigmoid instead of softmax**, 1×1 convs), on a
ZCU104 across all eight DPUCZDX8G sizes (B512–B4096).

Their reported cost: YOLOv8n VOC **0.60 → 0.45 mAP** after FPGA
deployment; YOLOv26n on B4096 at 34.05 FPS with a large quantization mAP
drop.

**Our route keeps attention exact and builds hardware for it — measured
accuracy cost: zero** (0.7360 identical, hardware vs software, same 664
detections, 400 images). This is the cleanest differentiation available
for the paper: *approximation-to-fit-the-DPU loses accuracy; an exact
custom accelerator beside the DPU does not.* Cite prominently.

Also found, not yet readable (paywalled, 403): **QYOLOv10** — a
quantization-aware, NMS-free YOLOv10 on **KV260** with a customized DPU
overlay (Integration VLSI Journal, 2025, S0167926025002482). Direct
same-board competitor for the comparison table — **fetch at college.**

---

## 4. Levers, ranked by payoff ÷ risk

### Tier A — no retraining, no new bitstream (software on the board)

**A1. Pipeline the real-video loop.** Apply `pipelined_throughput.py`'s
worker pattern (own `vart.Runner` per thread, shared LCAM CU behind a
lock) to `video_throughput_hybrid.py`. Decode/pre of frame N+1 overlaps
DPU of frame N. Ceiling ≈ the slowest serial resource (DPU ~43 ms, gate
lock ~27 ms) plus contention — plausibly **8.93 → ~15–19 FPS on real
video**, i.e. the 18-FPS headline becomes a *camera-to-boxes* number, not
only a static-tensor number. Highest-value, lowest-risk item on this list.
**Estimate, not measured — must be run on the board.**

**A2. Preprocess cost (25.8 ms).** Mostly `cv2.resize`/letterbox of
arbitrary-resolution frames (quantization already moved to a LUT, §42.5).
Worth profiling whether the letterbox canvas copy, not the resize, is the
cost, and whether moving it into the pipelined worker (A1) hides it
entirely — which is likely.

### Tier B — hardware / runtime engineering (bitstream or C++)

**B1. C++ VART zero-copy for the gate read-back (~21.5 ms).** Already
scoped and never attempted (`PROJECT_HISTORY.md` §37): Python's
`vart.TensorBuffer` exposes no physical address; the C++ API has
`data_phy()`. If the CU can write straight into the next DPU subgraph's
input buffer, the uncached read disappears. Largest single serial term
inside the gate call.

**B2. Fold head ops into the LCAM kernel invocation** (~2.7 ms CPU). Small
win; only worth doing alongside B1.

**B3. PL preprocessing via Vitis Vision (`resize` + quantize).** The
Kria smartcam app does exactly this. **Risk: CU budget** — hybrid3 is at
95.75% slices and an 8×32 FIFO change already failed timing. A second CU
very likely won't close. Only viable if A1 fails to hide preprocess.

**B4. Dual-core DPU — ruled out.** The anchor paper's KV260 2-core design
alone uses 72% LUT and 100% URAM; there is no room for the LCAM CU too.

### Tier C — training/model changes (need GPU; queue after v7)

**C1. Channel pruning of the v7 model (Vitis AI Optimizer, coarse-grained).**
Published YOLOX-S pruning: **−29% FLOPs with +1.8 mAP**, **−43% FLOPs with
only −0.2 mAP**. Directly cuts the 43 ms DPU term. Crucially, **the LCAM
HLS kernel takes H/W/C as runtime registers** (`H=0x10 W=0x18 C=0x20`), so
pruned channel counts need **no kernel change, no bitstream rebuild**.
Arguably a better first bet than v8's blunt width halving.

**C2. v8 lighter backbone (width 0.25) + knowledge distillation from v7.**
`build_v8_notebook.py` is built and verified, not pushed. Pair with
distillation (v7 as teacher): literature reports feature+logit KD
recovering most of the accuracy lost to shrinking (e.g. +2.5% relative
mAP50 on a nano student; up to ~50% size reduction without accuracy loss).
Note: no width-0.25 COCO checkpoint exists, so v8 alone leans on the
30-epoch schedule — KD matters more here than for C1.

**C3. Quantization-aware training.** The anchor paper loses 1.78 pts FP32→
INT8 (79.89→78.11). A KV260 YOLOX project found the **box-regression head
is the main INT8 casualty**. Vitis AI 3.0's PyTorch quantizer supports QAT.
Recovers accuracy at zero FPS cost.

**C4. Label cleaning.** Our own `--confusion` audit found fire-labelled-as-
smoke ground truth concentrated in scraped multi-source images. Relabel or
drop the audited-bad images, retrain, measure. Cheap; defensible;
mirrors SpaSE-UNet3D's audit contribution.

**C5. NMS-free head (YOLOv10-style one-to-one assignment).** Removes NMS;
QYOLOv10 did this on KV260. But `post` is only 7.3 ms here — low
priority for FPS, possibly useful for determinism.

### Tier D — architecture swaps (biggest lifts, lowest priority now)

- **YOLOv8/v10/v11 base**: DPUCZDX8G does not support SiLU — would need the
  same LeakyReLU swap we already did, plus a framework change. Vitis AI 3.0
  is the last release with a KV260 board image, which also caps tooling.
- **YOLO-FireAD AIR block** (1.45M params, Q/K/V + two sigmoid gates):
  kept as a stretch goal — a genuinely new accelerator, but a riskier HLS
  kernel with no evidence yet it's DPU-incompatible in a hardware-worthy
  way.

---

## 5. Recommended order

1. **A1 — pipelined real-video loop** (board only, no GPU, no rebuild).
   Turns the 18-FPS headline into a real-video number. Do first.
2. **Finish v7**, then **C1 pruning** of v7 (keeps LCAM + kernel intact),
   with **C3 QAT** on the pruned model before compiling.
3. **C2 v8 + KD** as the parallel "different architecture" data point, so
   the paper can show an accuracy-vs-DPU-latency Pareto (v7, v7-pruned,
   v8+KD), not one number.
4. **B1 C++ zero-copy** — the hardware contribution extension.
5. Stretch: AIR accelerator (Tier D).

Paper framing that falls out of this: *exact attention on a custom
accelerator beside the DPU (zero accuracy loss) vs. approximating
attention to fit the DPU (Karki et al.: large loss)*, plus a Pareto of
model sizes all using the same, unchanged accelerator.

---

## Sources

- [D-Fire dataset overview](https://www.emergentmind.com/topics/d-fire-dataset)
- [D-Fire: A dataset for fire and smoke object detection (MTAP)](https://dl.acm.org/doi/10.1007/s11042-022-13580-x)
- [Comparative analysis of YOLO models on D-Fire](https://www.researchgate.net/figure/Comparative-analysis-of-YOLO-models-on-D-Fire-dataset_tbl2_379851173)
- [Cross-dataset YOLOv8 fire/smoke evaluation, MDPI Drones](https://www.mdpi.com/2504-446X/10/8/635)
- [Karki et al., No Attention, No Problem (arXiv 2607.13106)](https://arxiv.org/html/2607.13106)
- [QYOLOv10 on KV260 (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0167926025002482)
- [KV260 YOLOX-Nano Vitis AI 3.0 project](https://github.com/carson-pol/KV260-yolox-vitis-ai)
- [Vitis AI 3.5 docs](https://xilinx.github.io/Vitis-AI/3.5/html/index.html)
- [Vitis AI 3.0 DPU IP & system integration](https://xilinx.github.io/Vitis-AI/3.0/html/docs/workflow-system-integration.html)
- [Kria smartcam accelerator HW architecture (Vitis Vision preprocess)](https://xilinx.github.io/kria-apps-docs/kv260/2022.1/build/html/docs/smartcamera/docs/hw_arch_accel.html)
- [Kria AI customization / DPU sizing](https://xilinx.github.io/kria-apps-docs/creating_applications/2022.1/build/html/docs/AI_customization.html)
- [Pruning YOLOv3 with Vitis AI on KV260](https://www.hackster.io/LogicTronix/pruning-yolov3-and-deploying-with-vitis-ai-on-kria-kv260-de654a)
- [Visual saliency-guided channel pruning (YOLOX-S FLOPs/mAP)](https://arxiv.org/pdf/2303.02512)
- [Knowledge distillation for lightweight weed detection](https://arxiv.org/pdf/2507.12344)
- [Ultralytics knowledge distillation guide](https://docs.ultralytics.com/guides/knowledge-distillation)
- [OwLite YOLO quantization (PTQ vs INT8)](https://blog.squeezebits.com/how-to-quantize-yolo-models-with-owlite-54076)
- [YOLOv10 NMS-free](https://docs.ultralytics.com/models/yolov10)
- [Sort-less FPGA NMS accelerator (IEEE)](https://ieeexplore.ieee.org/document/9634708/)
- [ECA-Net (IEEE)](https://ieeexplore.ieee.org/iel7/9142308/9156271/09156697.pdf)
