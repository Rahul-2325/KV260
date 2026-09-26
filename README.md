# LCAM-YOLOX on Kria KV260 — Hybrid DPU + Custom HLS Attention Accelerator

UAV wildfire (fire/smoke) detection on a Xilinx Kria KV260. The vendor DPU
cannot execute the LCAM attention gate natively, so on a DPU-only deployment
that step falls back to the ARM CPU. This project builds a dedicated Vitis HLS
accelerator for the gate and links it next to the DPU in one `v++ --link`
bitstream.

## Headline results (hardware-measured)

| Metric | Value |
|---|---|
| LCAM gate, CPU fallback → custom accelerator | 200.30 ms → 1.82 ms, bit-exact |
| End-to-end, single frame | 75.8 ms / 13.2 FPS (5.6x vs DPU+CPU baseline 2.34 FPS) |
| Pipelined throughput (4 workers) | 19.4 FPS |
| Real video, camera-to-boxes | 8.93 FPS (hardware gates) vs 3.06 FPS (software gates) |
| mAP@0.5 / mAP@.5:.95 | 0.7360 / 0.3774 — identical hardware vs software |
| Board power (INA260, measured) | 6.36 W; 0.440 J/frame (5.19x more efficient) |
| Resources (hybrid3) | 61,554 LUT · 81 BRAM · 50 URAM · 716 DSP · WNS +0.018 ns |

Full numbers and caveats: [`MY_RESULTS_SUMMARY.md`](MY_RESULTS_SUMMARY.md).
Complete chronological engineering log: [`PROJECT_HISTORY.md`](PROJECT_HISTORY.md).

## Repository layout

| Path | Contents |
|---|---|
| `hls_ip_sources/` | Vitis HLS C++ for the custom IPs (incl. `lcam_attention_gate`) |
| `vitis_hybrid/`, `vitis_hybrid_opt/` | Extensible platform + `v++ --link` flow for the DPU+LCAM hybrid; `package/` = board runtime (pipeline, mAP eval, video, power) |
| `vivado_scripts/` | Reconstructed Vivado Tcl (synthesis, implementation, JTAG/ILA) |
| `board_driver/`, `debug_scripts/` | Low-level IP driver and the diagnostic/benchmark scripts |
| `board_home_root/` | Scripts as they exist on the board (`/home/root`) |
| `training/` | Model training: LCAM module versions, fine-tune scripts (v3, v5), Vitis-AI quantization scripts, Kaggle notebooks |
| `accuracy_improvement/` | Current work: retraining plan, research directions, next-model notebooks |
| `vai25_yolo/` | 4K / tiled video experiments (stock DPU bitstream) |
| `results/`, `notes/`, `demo_output/` | Power/utilization reports, device-tree/xclbin notes, demo images and videos |
| `LITERATURE_REVIEW.md` | Related-work bibliography with links |

## Toolchain

Vitis HLS / Vivado / Vitis 2022.2 · Vitis AI 3.0 · XRT 2.14 · DPUCZDX8G B4096 ·
Kria KV260 (xck26-sfvc784-2LV-c).

## Not included

Paper PDFs (copyrighted — see links in `LITERATURE_REVIEW.md`), demo videos
over GitHub's 100 MB limit, trained `.pth` checkpoints, the D-Fire dataset,
and Vivado project/bitstream binaries (reproducible from the sources here).
The board SSH password is replaced with `<BOARD_PASSWORD>` throughout.
