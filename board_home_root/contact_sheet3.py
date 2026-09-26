"""
That sheet came back looking like the OLD static-camera clip, which is
suspicious. Two possible causes, tested separately here:
  1. cv2.VideoCapture's GStreamer backend on this board has already shown
     "Cannot query video position" warnings -- CAP_PROP_POS_FRAMES seeking
     may simply not work, silently returning frame 0 (or whatever is
     cached) for every requested index instead of actually seeking.
  2. The valset "photographs" might themselves be frame-grabs FROM fixed
     surveillance cameras (186/400 share the exact shape 1280x720, which is
     consistent with camera frame-grabs, not varied web photos), so many
     could be near-duplicate frames of a mostly-static scene.
Test (1) by reading SEQUENTIALLY (no .set()) instead of seeking.
Test (2) by reading the source JPGs directly, bypassing the video entirely.
"""
import cv2
import glob
import numpy as np

print("=== TEST 1: sequential read of the new video (no .set() seeking) ===")
cap = cv2.VideoCapture("/home/root/wildfire_real_720p.avi")
thumbs = []
idx = 0
target_idxs = set([0, 36, 72, 108, 145, 181, 217, 253, 290, 326, 362, 399])
while True:
    ok, f = cap.read()
    if not ok:
        break
    if idx in target_idxs:
        thumbs.append(cv2.resize(f, (320, 180)))
    idx += 1
cap.release()
print("sequential frames read:", idx)
rows = [np.hstack(thumbs[k:k+4]) for k in range(0, len(thumbs), 4)]
sheet = np.vstack([r for r in rows if r.shape[1] == 320 * 4])
cv2.imwrite("/home/root/sheet_seq.jpg", sheet)

print()
print("=== TEST 2: source JPGs directly, same 12 sorted indices ===")
files = sorted(glob.glob("/home/root/valset/images/*"))
print("total source files:", len(files))
thumbs2 = []
for i in sorted(target_idxs):
    img = cv2.imread(files[i])
    thumbs2.append((files[i], cv2.resize(img, (320, 180))))
for name, _ in thumbs2:
    print("  ", name)
rows2 = [np.hstack([t[1] for t in thumbs2[k:k+4]]) for k in range(0, len(thumbs2), 4)]
sheet2 = np.vstack([r for r in rows2 if r.shape[1] == 320 * 4])
cv2.imwrite("/home/root/sheet_src.jpg", sheet2)
