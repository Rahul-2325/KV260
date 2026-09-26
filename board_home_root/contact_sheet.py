import cv2
import numpy as np

for v, out in [("wildfire_720p.avi", "sheet_a.jpg"),
              ("wildfire_pos_720p.avi", "sheet_b.jpg"),
              ("lcam_wildfire_pos_det.avi", "sheet_c.jpg"),
              ("lcam_wildfire_720p_det.avi", "sheet_d.jpg")]:
    cap = cv2.VideoCapture("/home/root/vai25_models/" + v)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(n - 1, 0), 12, dtype=int)
    thumbs = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            continue
        thumbs.append(cv2.resize(f, (320, 180)))
    cap.release()
    if not thumbs:
        print(v, "NO FRAMES")
        continue
    rows = [np.hstack(thumbs[k:k+4]) for k in range(0, len(thumbs), 4)]
    rows = [r for r in rows if r.shape[1] == 320 * 4]
    sheet = np.vstack(rows) if rows else thumbs[0]
    cv2.imwrite("/home/root/" + out, sheet)
    print(v, "-> /home/root/" + out, "frames sampled:", list(idxs))
