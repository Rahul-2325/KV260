import cv2
for v in ["wildfire_720p.avi", "wildfire_pos_720p.avi", "lcam_wildfire_4k.avi",
         "lcam_wildfire_720p_det.avi", "lcam_wildfire_pos_det.avi"]:
    p = "/home/root/vai25_models/" + v
    cap = cv2.VideoCapture(p)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ok, frame = cap.read()
    mean = frame.mean() if ok else -1
    print("%-26s frames=%-5d %dx%d @%.1ffps  read_ok=%s  mean_px=%.1f"
          % (v, n, w, h, fps, ok, mean))
    cap.release()
