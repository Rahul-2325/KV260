# Literature Review — Working Bibliography for the LCAM-YOLOX Journal Paper

Compiled from web search (IEEE Xplore, Springer, MDPI, ACM, arXiv, PMC). Not
full-text extraction — titles/authors/venues/DOIs/links only, gathered so
they can be fed into NotebookLM alongside this project's own results
(`PROJECT_HISTORY.md`, `DEMO_RUNBOOK.md`, measured numbers) for drafting.

**Caveat**: entries came from search snippets, not always the paper itself.
Verify exact author lists/page numbers/venue names against the DOI link
before citing in the final manuscript — a few (marked ⚠) have lower
confidence on bibliographic detail (right paper, fuzzy metadata).

---

## 0. THE anchor paper — what this project extends

This is the model and baseline your whole project builds on. Confirmed by
you as the paper you are implementing/replicating: its DPU-only baseline
(195 FPS on KV260) matches the number in this project's own `PROJECT_CONTEXT.md`.

- **Elhence, A., Panda, A., Chamola, V., Sikdar, B.** (2025).
  "FPGA-Accelerated YOLOX With Enhanced Attention Mechanisms for
  Real-Time Wildfire Detection on AAVs." *IEEE Transactions on
  Instrumentation and Measurement*, vol. 74, pp. 1-14.
  DOI: [10.1109/TIM.2025.3556164](https://ieeexplore.ieee.org/document/10949830/)
  — Introduces LCAM-YOLOX (Layer-wise Channel Attention Module + YOLOX).
  KV260, DPU-only, INT8 quantized, 78.11% mAP, 195 FPS, 10.45 W.
  Authors affiliated with BITS Pilani and NUS.
  Author-hosted PDF mirror: https://www.ece.nus.edu.sg/stfpage/bsikdar/papers/tim_vinay_25.pdf

  **Positioning for your paper**: this paper's LCAM runs on the DPU's
  software/CPU fallback path (the DPU cannot execute the attention op
  natively). Your contribution is a dedicated HLS accelerator for that
  exact op, fused with the DPU in one hybrid bitstream — measured 1.82 ms
  bit-exact vs. ~200 ms CPU fallback for the same four gates. Frame your
  work explicitly as closing this paper's DPU/attention gap in hardware,
  not as a new detection model.

---

## 1. DPU / Vitis-AI architecture background

- AMD/Xilinx, "DPU IP Details and System Integration" — Vitis AI 3.0 / 3.5
  documentation (primary technical reference, not peer-reviewed):
  https://xilinx.github.io/Vitis-AI/3.0/html/docs/workflow-system-integration.html
  https://xilinx.github.io/Vitis-AI/3.5/html/docs/workflow-system-integration.html
- arXiv 2206.01981 (2022), "Evaluation of Xilinx Deep Learning Processing
  Unit under Neutron Irradiation." https://arxiv.org/pdf/2206.01981
  ⚠ tangential (radiation testing), but a real DPU-architecture citation.
- MDPI *Remote Sensing* 15(16):3975 (2023), "Edge Real-Time Object
  Detection and DPU-Based Hardware Implementation for Optical Remote
  Sensing Images." https://www.mdpi.com/2072-4292/15/16/3975

## 2. Kria KV260 platform benchmarking

- Springer, *Journal of Real-Time Image Processing* (2026), "Throughput
  impact of software multithreading for deep-learning inference on the
  AMD Kria KV260." https://link.springer.com/article/10.1007/s11554-026-01889-x
- arXiv 2606.12473 (2026), "Stereo Vision-Based Fall Prediction and
  Detection using Human Pose Estimation on the AMD Kria K26 SOM."
  https://arxiv.org/pdf/2606.12473
- arXiv 2507.16556 (2025), "Optimization of DNN-based HSI Segmentation
  FPGA-based SoC for ADS: A Practical Approach."
  https://arxiv.org/pdf/2507.16556
- Hackster.io, "Benchmarking the Kria KV260 AI Vision Starter Kit" —
  ⚠ not peer-reviewed, useful only as a practitioner data point, not a
  citable reference. https://www.hackster.io/whitney-knitter/benchmarking-the-kria-kv260-ai-vision-starter-kit-464972

## 3. Hybrid DPU + custom-accelerator co-design (architecturally closest prior art to your hybrid bitstream)

- **Kalantar, A., Schall, D., Zimmermann, R., et al.**, "FA-LAMP:
  FPGA-Accelerated Learned Approximate Matrix Profile for Time Series
  Similarity Prediction." *IEEE Conference* (FCCM 2021), document
  9443665. https://ieeexplore.ieee.org/document/9443665/
  Extended journal version: "FPGA-Based Acceleration of Time Series
  Similarity Prediction: From Cloud to Edge," *ACM Transactions on
  Reconfigurable Technology and Systems*, DOI:
  [10.1145/3555810](https://dl.acm.org/doi/10.1145/3555810).
  — Zynq UltraScale+ PS + Xilinx DPU IP + a custom HLS kernel implementing
  the GAP/FC/sigmoid layers the DPU can't run, both connected to DDR over
  AXI4. This is the closest architectural precedent found for "DPU +
  bespoke HLS kernel in one design" — cite as prior art for the general
  co-design pattern, then differentiate: FA-LAMP's custom kernel offloads
  generic dense/pooling layers, yours offloads a bespoke attention op and
  targets real-time vision instead of time-series.
  GitHub reference implementation: https://github.com/aminiok1/LAMP-FPGA

### 3b. Directly competing approaches (found 2026-09-27) — cite prominently

- **Karki, S., Ahmed, Q. A., Jungeblut, T.** (HSBI), "No Attention, No
  Problem: DPU-Aware Attention Approximation in Modern YOLO on FPGA,"
  arXiv 2607.13106, July 2026. https://arxiv.org/html/2607.13106
  — Same problem as ours (attention ops not DPU-native), OPPOSITE fix:
  approximate attention so it compiles onto the DPU (elementwise q⊙k for
  matmul, hard-sigmoid for softmax). ZCU104, all DPUCZDX8G sizes. Reports
  large accuracy cost (e.g. YOLOv8n VOC 0.60→0.45 mAP after deployment).
  **Our exact custom accelerator loses zero accuracy** — this is the key
  contrast for the paper.
- **QYOLOv10** — quantization-aware, NMS-free YOLOv10 on **KV260** with a
  customized DPU overlay, Integration (VLSI journal), 2025,
  https://www.sciencedirect.com/science/article/abs/pii/S0167926025002482
  — same board, direct competitor. ⚠ paywalled (403); fetch at college.

## 4. Attention mechanisms on FPGA / in CNNs

- Woo, S. et al. (2018), "CBAM: Convolutional Block Attention Module,"
  *ECCV 2018*. https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf
  — foundational channel/spatial attention reference; useful background
  for framing what LCAM is a variant of.
- IEEE document 10241832 (2023 conf.), "Performance Study of CBAM
  Attention Mechanism in Convolutional Neural Networks at Different
  Depths." https://ieeexplore.ieee.org/document/10241832/
- IEEE document 10806654, "Low-Bit Mixed-Precision Quantization and
  Acceleration of CNN for FPGA Deployment."
  https://ieeexplore.ieee.org/abstract/document/10806654
  ⚠ relevant to quantizing attention-bearing CNNs on FPGA, verify year.

## 5. YOLO-on-FPGA implementations (various YOLO versions, various boards)

- PMC12568313, "Design and Implementation of a YOLOv2 Accelerator on a
  Zynq-7000 FPGA." https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12568313/
- arXiv 2605.06745 (2026), "Development of embedded target detection
  system based on FPGA and YOLOv3-Tiny." https://arxiv.org/pdf/2605.06745
- IEEE document 9817206 (2022 conf.), "An Evaluation and Embedded
  Hardware Implementation of YOLO for Real-Time Wildfire Detection."
  https://ieeexplore.ieee.org/document/9817206/ — directly on-topic
  (wildfire + YOLO + embedded hardware), predates the LCAM-YOLOX paper.
- MDPI *Electronics* 15(11):2442, "An Energy-Efficient FPGA-Based CNN
  Accelerator with Dual-Multiply Packing and Ping-Pong Buffering for
  Real-Time Object Detection." https://doi.org/10.3390/electronics15112442
- PMC11434529, "An FPGA-Based YOLOv5 Accelerator for Real-Time Industrial
  Vision Applications." https://pmc.ncbi.nlm.nih.gov/articles/PMC11434529/
- arXiv 2503.13023 (2025), "Real-Time Multi-Object Tracking using YOLOv8
  and SORT on a SoC FPGA." https://arxiv.org/abs/2503.13023
- ResearchGate 392363887, "Performance Optimization of FPGA-Accelerated
  YOLOv8 for Driver Drowsiness Detection Using Vivado HLS."
  https://www.researchgate.net/publication/392363887 ⚠ verify venue/year.
- arXiv 2507.18174 (2025), "Real-Time Object Detection and Classification
  using YOLO for Edge FPGAs." https://arxiv.org/pdf/2507.18174
- IEEE document 10497598, "Rapid Detection of PCB Defects Based on
  YOLOx-Plus and FPGA." https://ieeexplore.ieee.org/document/10497598/
  — different application domain, but a second YOLOX-on-FPGA precedent
  worth noting for the YOLOX-specific hardware-mapping discussion.

## 6. Wildfire / fire / smoke detection — algorithmic + edge deployment (not necessarily FPGA)

- PMC11085648, "Enhanced Lightweight YOLOX for Small Object Wildfire
  Detection in UAV Imagery." https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11085648/
- arXiv 2401.08105 (2024), "Hardware Acceleration for Real-Time Wildfire
  Detection Onboard Drone Networks." https://arxiv.org/pdf/2401.08105
  — highly relevant, same problem framing (UAV + hardware accel + wildfire).
- arXiv 2309.01318 (2023), drone wildfire imagery segmentation on an FPGA
  smart camera (BNN + U-Net, Ultra96-v2). https://arxiv.org/pdf/2309.01318
- arXiv 2409.12635 (2024), "EFA-YOLO: An Efficient Feature Attention
  Model for Fire and Flame Detection." https://arxiv.org/pdf/2409.12635
  — another attention+YOLO+fire combination, good direct comparator.
- arXiv 2505.20884 (2025), "YOLO-FireAD: Efficient Fire Detection via
  Attention-Guided Inverted Residual Learning and Dual-Pooling Feature
  Preservation." https://arxiv.org/pdf/2505.20884
- MDPI *Drones* 8(9):483 (2024), "Real-Time Fire Detection: Integrating
  Lightweight Deep Learning Models on Drones with Edge Computing."
  https://www.mdpi.com/2504-446X/8/9/483
- MDPI *Sensors* 25(20):6419 (2025), "Edge-Based Autonomous Fire and
  Smoke Detection Using MobileNetV2." https://doi.org/10.3390/s25206419
- PMC13210558 (2025), "Edge-Friendly UAV Wildfire Smoke and Flame
  Detection Using Transfer Learning-Enhanced Lightweight Deep Learning
  Models." https://pmc.ncbi.nlm.nih.gov/articles/PMC13210558/
- Springer, *Fire Technology* (2026), "Multi-core Edge Computing with
  Deep Neural Networks for Real-Time Fire Detection."
  https://link.springer.com/article/10.1007/s10694-026-01911-5
- Springer, *Signal, Image and Video Processing* (2026), "Lightweight
  SFGI-YOLO for real-time forest fire and smoke detection from images on
  Raspberry Pi edge devices." https://link.springer.com/article/10.1007/s11760-026-05686-8
- MDPI *Applied Sciences* 16(2):778 (2026), "ESCFM-YOLO: Lightweight
  Dual-Stream Architecture for Real-Time Small-Scale Fire Smoke Detection
  on Edge Devices." https://www.mdpi.com/2076-3417/16/2/778
- PLOS ONE (2025), "FCMI-YOLO: An efficient deep learning-based algorithm
  for real-time fire detection on edge devices."
  https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0329555
- *Scientific Reports* (2024), "YOLOFM: an improved fire and smoke object
  detection algorithm based on YOLOv5n." https://www.nature.com/articles/s41598-024-55232-0

## 7. Quantization / general FPGA-CNN-accelerator surveys

- arXiv 2509.04153 (2025), "Real Time FPGA Based CNNs for Detection,
  Classification, and Tracking in Autonomous Systems: State of the Art
  Designs and Optimizations." https://arxiv.org/pdf/2509.04153
  — broadest, most directly relevant survey found; good "related work"
  backbone citation for the whole FPGA-CNN section of the introduction.
- arXiv 2412.15666 (2024), "A survey on FPGA-based accelerator for ML
  models." https://arxiv.org/pdf/2412.15666
- arXiv 2505.13461 (2025), "FPGA-based Acceleration for Convolutional
  Neural Networks: A Comprehensive Review." https://arxiv.org/pdf/2505.13461
- MDPI *Applied Sciences* 15(2):688 (2025), "FPGA-QNN: Quantized Neural
  Network Hardware Acceleration on FPGAs." https://doi.org/10.3390/app15020688
- ScienceDirect (2025), "Post-training quantization for efficient
  FPGA-based neural network acceleration."
  https://www.sciencedirect.com/science/article/abs/pii/S0167926025001658
- arXiv 2508.21493 (2025), "SIRA: Scaled-Integer Range Analysis for
  Optimizing FPGA Dataflow Neural Network Accelerators."
  https://arxiv.org/pdf/2508.21493

---

## Suggested year-wise skeleton (for the paper's Related Work section)

- **2018**: CBAM (attention foundation, §4)
- **2021**: FA-LAMP FCCM paper (§3, closest hybrid DPU+HLS precedent)
- **2022**: YOLO-wildfire embedded eval (§6, IEEE 9817206); DPU neutron
  eval (§1); ACM TRETS FA-LAMP journal version (§3)
- **2023**: CBAM depth study (§4); DPU-based remote-sensing detector
  (§1); FPGA smart-camera wildfire segmentation (§6, arXiv 2309.01318)
- **2024**: EFA-YOLO (§6); wildfire hardware-accel drone survey (§6,
  arXiv 2401.08105); YOLOFM (§6); MDPI Drones fire-on-drones (§6); FPGA
  survey (§7, arXiv 2412.15666)
- **2025**: **LCAM-YOLOX anchor paper (§0)**; YOLO-FireAD (§6); MDPI
  Sensors edge fire detection (§6); PMC edge-friendly UAV wildfire (§6);
  FCMI-YOLO (§6); FPGA-QNN (§7); PTQ FPGA (§7); SIRA (§7); comprehensive
  FPGA-CNN review (§7); broad autonomous-systems FPGA-CNN survey (§7,
  arXiv 2509.04153); YOLOv8 SoC-FPGA tracking (§5)
- **2026**: multiple Kria KV260 benchmarking papers (§2); several
  lightweight fire/smoke edge papers (§6, SFGI-YOLO, ESCFM-YOLO,
  Fire Technology multi-core); FPGA+YOLOv3-Tiny embedded system (§5)

---

## Gaps to fill in a follow-up pass (tell me if you want these run)

- Exact author/venue metadata for the ⚠-flagged entries above.
- A dedicated search for other papers CITING the LCAM-YOLOX anchor paper
  (Google Scholar "cited by" for DOI 10.1109/TIM.2025.3556164) — these
  would be the most important direct competitors/successors to discuss.
- Springer-specific and ACM-specific venue searches (this pass leaned
  IEEE/arXiv/MDPI-heavy) if you want tighter Springer coverage.
- Full-text pull of the anchor paper itself (I could not get past IEEE's
  paywall/403 with WebFetch; the author-hosted NUS PDF mirror linked in
  §0 may be open — worth trying directly, or ask your library's
  IEEE Xplore access) so we can compare its exact resource utilization
  and pipeline-stage timing tables against this project's own numbers.
