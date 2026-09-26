"""
Side-by-side check on one image: ground truth | v5 (board, INT8, last week) | v7 (FP32).
Writes results_v7/single_image/compare_<name>.jpg and prints IoU of each prediction vs GT.

v5 boxes are the board's hybrid-pipeline output for WEB09971.jpg recorded in
PROJECT_HISTORY.md (sections 35.3 / 41), with class names corrected
(see CORRECTION_CLASS_LABELS.md: the board printed class 0 as "fire").
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluate_model import CLASSES, COLORS, build_model, predict, load_gt, box_iou  # noqa: E402

IMG = os.path.join(HERE, "results_v7/single_image/WEB09971.jpg")
LBL = os.path.expanduser("~/wildfire_project/dataset/extracted/data/test/labels/WEB09971.txt")
CKPT = os.path.expanduser("~/wildfire_project/kaggle_v7_output/final_models/lcam_yolox_wildfire_v7_BEST.pth")
V5_BOARD = [  # (class_id, score, box)
    (0, 0.875, [378, 5, 769, 477]),
    (1, 0.651, [469, 418, 713, 494]),
    (1, 0.301, [607, 513, 634, 541]),
]
CONF = 0.25


def panel(img, boxes, title, gt=False):
    vis = img.copy()
    t = max(2, img.shape[1] // 300)
    for c, s, b in boxes:
        col = (0, 200, 0) if gt else COLORS[c]
        p1, p2 = (int(b[0]), int(b[1])), (int(b[2]), int(b[3]))
        cv2.rectangle(vis, p1, p2, col, t)
        lab = CLASSES[c] if gt else f"{CLASSES[c]} {s:.2f}"
        cv2.putText(vis, lab, (p1[0] + 4, max(p1[1] + 22, 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(vis, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return vis


def report(name, dets, gb, gc):
    print(f"\n{name}")
    for c, s, b in dets:
        same = gb[gc == c]
        iou = box_iou(np.array([b], float), same).max() if len(same) else 0.0
        print(f"  {CLASSES[c]:5s} {s:.3f}  box={list(map(int, b))}  IoU vs GT {CLASSES[c]} = {iou:.3f}")


def main():
    img = cv2.imread(IMG)
    h, w = img.shape[:2]
    gb, gc = load_gt(LBL, w, h)
    model = build_model(CKPT)
    boxes, scores, cls = predict(model, img)
    v7 = [(int(c), float(s), b) for b, s, c in zip(boxes, scores, cls) if s >= CONF]
    gt = [(int(c), 1.0, b) for b, c in zip(gb, gc)]

    print(f"image {w}x{h}")
    print("ground truth:", [(CLASSES[c], list(map(int, b))) for c, _, b in gt])
    report("v5 board (INT8, hybrid accelerator):", V5_BOARD, gb, gc)
    report("v7 (FP32, PyTorch):", v7, gb, gc)

    row = cv2.hconcat([panel(img, gt, "Ground truth", gt=True),
                       panel(img, V5_BOARD, "v5 board INT8 (last week)"),
                       panel(img, v7, "v7 FP32 (new)")])
    out = os.path.join(HERE, "results_v7/single_image/compare_WEB09971.jpg")
    cv2.imwrite(out, row)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
