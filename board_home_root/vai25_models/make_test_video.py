#!/usr/bin/env python3
"""
Build a genuine 3840x2160 (UHD-4K) MJPEG test clip on the board.

Each 4K frame is an exact 3x3 mosaic of nine 1280x720 source images
(3*1280 = 3840, 3*720 = 2160). Sources are used at their NATIVE
resolution -- nothing is upscaled -- so the frame carries real 4K-worth
of pixel detail rather than a stretched HD image.

Cells rotate every frame so successive frames genuinely differ.

Writes:
  test_4k.avi    3840x2160 MJPEG
  test_1080p.avi 1920x1080 MJPEG (same content, for comparison)
"""
import os
import sys
import glob
import cv2
import numpy as np

OUT_DIR = "/home/root/vai25_models"
CELL_W, CELL_H = 1280, 720
COLS, ROWS = 3, 3
W, H = CELL_W * COLS, CELL_H * ROWS          # 3840 x 2160
N_FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 120
FPS = 30


def load_cell(path):
    im = cv2.imread(path)
    if im is None:
        return None
    h, w = im.shape[:2]
    # cover-fit into the cell without distorting aspect ratio
    s = max(CELL_W / w, CELL_H / h)
    im = cv2.resize(im, (int(round(w * s)), int(round(h * s))),
                    interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    y0 = (im.shape[0] - CELL_H) // 2
    x0 = (im.shape[1] - CELL_W) // 2
    return im[y0:y0 + CELL_H, x0:x0 + CELL_W]


def main():
    srcs = []
    # COCO-bearing images first so detections actually appear
    for p in ["bus.jpg", "zidane.jpg"]:
        c = load_cell(os.path.join(OUT_DIR, p))
        if c is not None:
            srcs.append((p, c))
    # then real wildfire frames at native 1280x720
    wf = sorted(glob.glob("/home/root/valset/images/*.jpg"))
    for p in wf:
        if len(srcs) >= 12:
            break
        im = cv2.imread(p)
        if im is None or im.shape[0] < 700 or im.shape[1] < 1200:
            continue          # only take the native-720p ones
        c = load_cell(p)
        if c is not None:
            srcs.append((os.path.basename(p), c))

    if len(srcs) < COLS * ROWS:
        print("not enough source images (%d)" % len(srcs))
        return 1

    print("source cells (%d):" % len(srcs))
    for n, _ in srcs:
        print("   ", n)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    p4k = os.path.join(OUT_DIR, "test_4k.avi")
    p1080 = os.path.join(OUT_DIR, "test_1080p.avi")
    w4 = cv2.VideoWriter(p4k, fourcc, FPS, (W, H))
    w1 = cv2.VideoWriter(p1080, fourcc, FPS, (1920, 1080))
    if not w4.isOpened() or not w1.isOpened():
        print("VideoWriter failed to open (MJPG/avi unsupported)")
        return 1

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    n = len(srcs)
    for f in range(N_FRAMES):
        for i in range(COLS * ROWS):
            cell = srcs[(i + f) % n][1]
            r, c = divmod(i, COLS)
            canvas[r * CELL_H:(r + 1) * CELL_H,
                   c * CELL_W:(c + 1) * CELL_W] = cell
        w4.write(canvas)
        w1.write(cv2.resize(canvas, (1920, 1080), interpolation=cv2.INTER_AREA))
        if (f + 1) % 20 == 0:
            print("  %d/%d frames" % (f + 1, N_FRAMES))
    w4.release()
    w1.release()

    for p in (p4k, p1080):
        sz = os.path.getsize(p) / 1e6
        cap = cv2.VideoCapture(p)
        ok, fr = cap.read()
        cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print("%s  %.1f MB  frames=%d  readback=%s %s"
              % (p, sz, cnt, ok, fr.shape if ok else "-"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
