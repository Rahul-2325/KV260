import cv2
import numpy as np
import json

manifest = json.load(open("/home/root/wildfire_real_720p.json"))
# frames known to have ground-truth objects (from labels), pick a spread
pos_frames = []
import os
for m in manifest:
    stem = os.path.splitext(m["source"])[0]
    lp = "/home/root/valset/labels/%s.txt" % stem
    if os.path.exists(lp) and open(lp).read().strip():
        pos_frames.append(m["frame"])
print("positive frames available:", len(pos_frames))
pick = pos_frames[::max(1, len(pos_frames)//8)][:8]
print("picking frames:", pick)

cap = cv2.VideoCapture("/home/root/wildfire_real_720p_det.avi")
thumbs = []
idx = 0
target = set(pick)
while True:
    ok, f = cap.read()
    if not ok:
        break
    if idx in target:
        thumbs.append(cv2.resize(f, (320, 180)))
    idx += 1
cap.release()
rows = [np.hstack(thumbs[k:k+4]) for k in range(0, len(thumbs), 4)]
sheet = np.vstack([r for r in rows if r.shape[1] == 320*4])
cv2.imwrite("/home/root/sheet_det.jpg", sheet)
print("wrote sheet with", len(thumbs), "annotated positive frames")
