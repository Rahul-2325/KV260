# accuracy_improvement/ — new work (from 2026-09-26)

All new results, corrections and plans live here. The original project
documents in the parent folder are kept unchanged.

| File | What it is |
|---|---|
| `SESSION_2026-09-27.md` | **Start here** — what was done, where things are, next steps |
| `compare_single_image.py` | Ground truth vs v5 (board) vs v7 on one image |
| `PLAN.md` | Running plan: strategic redirect, v7 result, FPS levers, candidate architectures, next steps |
| `RESEARCH_DIRECTIONS.md` | Cross-layer survey (model, training, quantization, DPU, runtime, PL) ranked against the measured frame-time breakdown; key competing papers |
| `CORRECTION_CLASS_LABELS.md` | **Read first when quoting old per-class or confusion results** — board scripts had smoke/fire names swapped |
| `evaluate_model.py` | Paper-style evaluation: COCO mAP, per-class AP, P/R/F1, confusion matrix, PR/F1/P/R curves, params/GFLOPs/latency, sample predictions; also single-image prediction (`--image`) |
| `results_v7/` | Output of `evaluate_model.py` for the v7 model (report.md, metrics.json, figures) |
| `finetune_lcam_v6.py` | Improved fine-tune script (superseded by the v7 full retrain) |
| `build_v8_notebook.py`, `lcam_v8_lightweight.ipynb`, `kernel-metadata.json` | v8 lighter-backbone (width 0.25) Kaggle notebook — built, not run |

## Models

| Model | What | mAP@0.5 | mAP@.5:.95 | Where measured |
|---|---|---|---|---|
| v5 (deployed on KV260) | hand-rolled fine-tune | 0.7360 | 0.3774 | board, INT8, 400-img val slice |
| v3.1 (old notebook) | proper YOLOX recipe, sigmoid LCAM | 0.765 | 0.430 | Kaggle, FP32, full val |
| **v7** | proper YOLOX recipe, v5 hard-sigmoid LCAM, COCO-pretrained | **0.775** | **0.435** | FP32, full val (3,099) |
| **v7** | same model | **0.766** | **0.419** | FP32, **held-out test (4,306)** |

Full v7 evaluation (val + held-out test): `results_v7/report.md`.

## v7 held-out test results (paper table)

| Model | Params | GFLOPs | P | R | F1 | mAP@0.5 | mAP@0.75 | mAP@.5:.95 |
|---|---|---|---|---|---|---|---|---|
| LCAM-YOLOX v7 | 8.98 M | 26.25 | 0.688 | 0.769 | 0.726 | 0.766 | 0.403 | 0.419 |

P/R/F1 at conf 0.25; best mean F1 0.746 at conf 0.434 (P 0.793, R 0.704).

| Class | GT boxes | AP@0.5 | AP@.5:.95 |
|---|---|---|---|
| smoke | 2,311 | 0.825 | 0.478 |
| fire | 2,878 | 0.707 | 0.360 |

By object size (COCO): AP_small 0.195, AP_medium 0.382, AP_large 0.501.
Smoke↔fire confusion: 25 of 3,983 matched detections (0.6%). Main error
sources: missed objects (410 smoke, 796 fire) and false positives
(591 smoke, 1,277 fire) at conf 0.25.

Single-image check (`WEB09971.jpg`, same image as last week's board run):
`results_v7/single_image/compare_WEB09971.jpg` — v7 finds smoke 0.795 and
fire 0.712 with no false positive; v5 on the board had one extra false
fire (0.301) on a street light.
