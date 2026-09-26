# Evaluation report — `lcam_yolox_wildfire_v7_BEST.pth` (FP32, PyTorch)

Classes: 0 = smoke, 1 = fire. mAP uses conf>0.001, NMS 0.65 (training exp settings); P/R/F1 and confusion use conf=0.25 unless noted, IoU 0.5.

## Model complexity

| Params (M) | GFLOPs @640 | CPU FP32 latency (ms) | CPU threads |
|---|---|---|---|
| 8.98 | 26.25 | 177.4 | 16 |

CPU latency is a PC reference only — not the KV260/DPU figure.

## Headline table

| Split | Images | P | R | F1 | mAP@0.5 | mAP@0.75 | mAP@0.5:0.95 |
|---|---|---|---|---|---|---|---|
| val | 3099 | 0.685 | 0.771 | 0.725 | 0.775 | 0.428 | 0.435 |
| test | 4306 | 0.688 | 0.769 | 0.726 | 0.766 | 0.403 | 0.419 |

## val split

### Per-class

| Class | GT boxes | AP@0.5 | AP@0.5:0.95 | P@0.25 | R@0.25 | F1@0.25 |
|---|---|---|---|---|---|---|
| smoke | 1755 | 0.830 | 0.493 | 0.746 | 0.814 | 0.779 |
| fire | 2176 | 0.721 | 0.378 | 0.623 | 0.728 | 0.671 |

Best mean F1 = 0.750 at conf = 0.463 (P 0.811, R 0.698).

### COCO breakdown

| AP | AP50 | AP75 | AP_S | AP_M | AP_L | AR_1 | AR_10 | AR_100 |
|---|---|---|---|---|---|---|---|---|
| 0.435 | 0.775 | 0.428 | 0.208 | 0.385 | 0.527 | 0.352 | 0.544 | 0.573 |

### Confusion matrix (rows = predicted, cols = true)

| | smoke | fire | background |
|---|---|---|---|
| **smoke** | 1427 | 17 | 470 |
| **fire** | 4 | 1581 | 958 |
| **background** | 324 | 578 | 0 |

Figures: `val_pr_curve.png`, `val_f1_curve.png`, `val_p_curve.png`, `val_r_curve.png`, `val_confusion_matrix*.png`, `val_sample_predictions.jpg`

## test split

### Per-class

| Class | GT boxes | AP@0.5 | AP@0.5:0.95 | P@0.25 | R@0.25 | F1@0.25 |
|---|---|---|---|---|---|---|
| smoke | 2311 | 0.825 | 0.478 | 0.759 | 0.820 | 0.788 |
| fire | 2878 | 0.707 | 0.360 | 0.616 | 0.718 | 0.663 |

Best mean F1 = 0.746 at conf = 0.434 (P 0.793, R 0.704).

### COCO breakdown

| AP | AP50 | AP75 | AP_S | AP_M | AP_L | AR_1 | AR_10 | AR_100 |
|---|---|---|---|---|---|---|---|---|
| 0.419 | 0.766 | 0.403 | 0.195 | 0.382 | 0.501 | 0.343 | 0.531 | 0.560 |

### Confusion matrix (rows = predicted, cols = true)

| | smoke | fire | background |
|---|---|---|---|
| **smoke** | 1891 | 15 | 591 |
| **fire** | 10 | 2067 | 1277 |
| **background** | 410 | 796 | 0 |

Figures: `test_pr_curve.png`, `test_f1_curve.png`, `test_p_curve.png`, `test_r_curve.png`, `test_confusion_matrix*.png`, `test_sample_predictions.jpg`
