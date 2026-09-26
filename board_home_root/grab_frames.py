import cv2
for v, out, idx in [("wildfire_720p.avi", "frame_a.jpg", 60),
                    ("wildfire_pos_720p.avi", "frame_b.jpg", 60)]:
    cap = cv2.VideoCapture("/home/root/vai25_models/" + v)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if ok:
        cv2.imwrite("/home/root/" + out, frame)
        print(v, "-> /home/root/" + out, frame.shape, "mean=%.1f" % frame.mean())
    else:
        print(v, "FAILED to read frame", idx)
    cap.release()
