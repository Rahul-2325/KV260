"""Are the first 40 sorted images genuinely low/no-object, or is 0 detections
suspicious? Check ground truth directly."""
import json
import os

manifest = json.load(open("/home/root/wildfire_real_720p.json"))
n_obj_total = 0
n_frames_with_obj = 0
for m in manifest[:40]:
    stem = os.path.splitext(m["source"])[0]
    lp = "/home/root/valset/labels/%s.txt" % stem
    n = 0
    if os.path.exists(lp):
        with open(lp) as f:
            n = sum(1 for line in f if line.strip())
    n_obj_total += n
    if n > 0:
        n_frames_with_obj += 1
    print("frame %3d  %-28s  gt_objects=%d" % (m["frame"], m["source"], n))
print()
print("first 40 frames: %d ground-truth objects across %d/40 frames"
      % (n_obj_total, n_frames_with_obj))

# and the full 400, for reference
n_obj_all = 0
n_frames_all = 0
for m in manifest:
    stem = os.path.splitext(m["source"])[0]
    lp = "/home/root/valset/labels/%s.txt" % stem
    n = 0
    if os.path.exists(lp):
        with open(lp) as f:
            n = sum(1 for line in f if line.strip())
    n_obj_all += n
    if n > 0:
        n_frames_all += 1
print("all 400 frames : %d ground-truth objects across %d/400 frames"
      % (n_obj_all, n_frames_all))
