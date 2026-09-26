#!/usr/bin/env python3
"""
build_wildfire_video.py -- run ON THE BOARD.

Builds a REAL fire/smoke test video from the 400 labelled images already
used for the mAP=0.7360 result (eval_map.py), because no genuine fire/smoke
VIDEO exists anywhere in this project. The two files that looked like one
(wildfire_720p.avi, wildfire_pos_720p.avi) turned out, on inspection with a
12-frame contact sheet spanning each whole clip, to be a single static
surveillance frame looped -- identical timestamp overlay in every sampled
frame, no fire or smoke visible at any point. Confirmed by the user, not
just a guess.

This is the SAME METHOD already used elsewhere in this project for a
compositional test video (4K_VIDEO_RESEARCH.md section 6.B's mosaic clips):
real photographs, sequenced into a video container, rather than synthetic
frames. Unlike that COCO-domain mosaic, every source image here is genuine
fire/smoke content WITH YOLO ground truth (valset/labels/), so real
detections are expected, not the "correct negative" result Track B got on
its COCO models.

Source images vary wildly in native resolution (400 images span >90
distinct shapes, from 240x320 to 1920x1080). Each is letterboxed into a
FIXED 1280x720 canvas (grey pad, value 114 -- the same convention the model
itself uses for its own 640x640 letterbox) rather than stretched, so aspect
ratio and content are preserved exactly as in the original photo.

1280x720 was chosen because it is the single most common native resolution
in this dataset (186 of 400 images) AND the model's own training image
resolution (see PROJECT_HISTORY 4K_VIDEO_RESEARCH.md section 12A) -- so
frames that are already 720p pass through with no resampling at all.

Writes a frame-index -> source-filename manifest alongside the video, so a
later script can look up ground truth for any given frame without
re-deriving the mapping.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np


def letterbox_canvas(img, w=1280, h=720):
    oh, ow = img.shape[:2]
    s = min(w / ow, h / oh)
    nw, nh = int(round(ow * s)), int(round(oh * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((h, w, 3), 114, dtype=np.uint8)
    px, py = (w - nw) // 2, (h - nh) // 2
    canvas[py:py + nh, px:px + nw] = r
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="/home/root/valset/images")
    ap.add_argument("--out", default="/home/root/wildfire_real_720p.avi")
    ap.add_argument("--manifest", default="/home/root/wildfire_real_720p.json")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all images")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.images, "*")))
    if args.limit:
        files = files[:args.limit]
    print("source images: %d" % len(files))

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (1280, 720))

    manifest = []
    written = 0
    for i, f in enumerate(files):
        img = cv2.imread(f)
        if img is None:
            print("  SKIP (unreadable): %s" % f)
            continue
        canvas = letterbox_canvas(img)
        writer.write(canvas)
        manifest.append({"frame": written, "source": os.path.basename(f),
                         "orig_shape": list(img.shape)})
        written += 1
        if written % 100 == 0:
            print("  %d / %d written" % (written, len(files)))

    writer.release()
    with open(args.manifest, "w") as fh:
        json.dump(manifest, fh, indent=1)

    print()
    print("wrote %d frames -> %s  (%.1f s @ %.1f fps)"
          % (written, args.out, written / args.fps, args.fps))
    print("manifest (frame -> source image, for ground-truth lookup) -> %s"
          % args.manifest)
    print()
    print("size on disk: %.1f MB" % (os.path.getsize(args.out) / 1e6))


if __name__ == "__main__":
    main()
