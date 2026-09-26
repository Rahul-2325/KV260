# Correction — smoke/fire class names were swapped in the board scripts

Found 2026-09-27. The original documents are left unchanged on purpose;
this file records the correct reading of the affected results.

## What was wrong

D-Fire (and our training code, every Kaggle notebook, the COCO conversion)
uses **class 0 = smoke, class 1 = fire**. The board-side scripts defaulted
to the opposite name order (`--classes fire,smoke`), so every class NAME
they printed or drew was swapped:

- `vitis_hybrid/package/eval_map.py`, `hybrid_pipeline.py`,
  `video_throughput_hybrid.py`
- `debug_scripts/postprocess_detections.py`
- `vai25_yolo/lcam_video_4k.py`, `vai25_yolo/tiled_lcam.py`
- `notes/infer_kv260.py`

**The numbers were always measured correctly** — detections, boxes, scores,
mAP, bit-exactness, FPS are all unaffected. Only which label text was
attached to class 0 vs class 1 was wrong.

## How it was verified

Rendered ground-truth-only contact sheets from the 3,099-image val split:
images containing only class-0 boxes are all smoke plumes; images containing
only class-1 boxes are all flames (campfire, burning vehicle, night-time
fire spots). Val box counts: class 0 = 1,756, class 1 = 2,176.

## Corrected reading of the affected results

**Per-class AP on the board** (`MY_RESULTS_SUMMARY.md` §5 /
`PROJECT_HISTORY.md` §38.1), 400-image slice, INT8, hardware and software
gates identical:

| Class | Objects | AP@0.5 | AP@.5:.95 |
|---|---|---|---|
| **smoke** (id 0) | 208 | 0.7378 | 0.4103 |
| **fire** (id 1) | 292 | 0.7342 | 0.3446 |
| mAP | | 0.7360 | 0.3774 |

(The old table listed these two rows under the opposite names.)

**Class-confusion analysis** (`MY_RESULTS_SUMMARY.md` §8.3 /
`PROJECT_HISTORY.md` §42.7bis), conf 0.30:

| Predicted → actual | Count | |
|---|---|---|
| smoke → smoke | 152 | correct |
| **smoke → fire** | **2** | **confused: model said smoke, ground truth is fire** |
| fire → fire | 208 | correct |
| no matching ground truth | 104 | hallucinations |

**The conclusion about the two confusion cases is reversed.** `WEB03809`
and `PublicDataset01011` are blazing-fire images whose ground truth is
class 1 = **fire** — the labels are **correct**. The model called them
**smoke**. These are genuine (rare, 0.6%) model errors, **not** dataset
labelling errors as the old text concluded. For the paper: report 0.6% as
a real, small model limitation.

**The demo video's "fire 0.79" on a distant smoke plume** was a class-0
detection — i.e. the model correctly said **smoke**; only the on-screen
text was wrong.

## Status

- Code: all seven scripts above now default to `smoke,fire` (the only
  change in each file, verified by diff against the GitHub copy).
- Board: the copies in `/home/root/hybrid_pkg/` on the KV260 are still the
  old ones and need re-copying before the next board run.
- The new PC-side evaluation (`evaluate_model.py`) uses the correct names.
