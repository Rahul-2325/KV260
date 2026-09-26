import cv2
import numpy as np

cap = cv2.VideoCapture("/home/root/wildfire_real_720p.avi")
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
idxs = np.linspace(0, max(n - 1, 0), 12, dtype=int)
thumbs = []
for i in idxs:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    ok, f = cap.read()
    if ok:
        thumbs.append(cv2.resize(f, (320, 180)))
cap.release()
rows = [np.hstack(thumbs[k:k+4]) for k in range(0, len(thumbs), 4)]
sheet = np.vstack([r for r in rows if r.shape[1] == 320 * 4])
cv2.imwrite("/home/root/sheet_real.jpg", sheet)
print("frames sampled:", list(idxs), " total:", n)
