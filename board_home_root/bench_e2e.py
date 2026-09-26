import xir, vitis_ai_library
import numpy as np
import cv2
import time

XMODEL = '/workspace/compiled_v5/lcam_v5.xmodel'
IMG_PATH = '/home/root/WEB09971.jpg'

# ---- Load model ----
graph = xir.Graph.deserialize(XMODEL)
runner = vitis_ai_library.GraphRunner.create_graph_runner(graph)
input_tensors = runner.get_input_tensors()
output_tensors = runner.get_output_tensors()
print("Input shape:", tuple(input_tensors[0].dims))
print("Output shape:", [tuple(t.dims) for t in output_tensors])

# ---- Preprocess (confirmed NHWC from xdputil dump, fixpos=-1 float input) ----
img0 = cv2.imread(IMG_PATH)
img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
h0, w0 = img0.shape[:2]
r = min(640/h0, 640/w0)
rimg = cv2.resize(img0, (int(w0*r), int(h0*r)))
canvas = np.full((640, 640, 3), 114, np.uint8)
canvas[:rimg.shape[0], :rimg.shape[1]] = rimg
input_data = canvas.astype(np.float32)[np.newaxis, ...]

# ---- Warm-up ----
print("Warming up...")
for _ in range(10):
    out_list = [np.empty(tuple(t.dims), dtype=np.float32) for t in output_tensors]
    job_id = runner.execute_async([input_data], out_list)
    runner.wait(job_id)
print("Warm-up done")

# ---- Steady-state timed benchmark ----
N = 100
out_list = [np.empty(tuple(t.dims), dtype=np.float32) for t in output_tensors]
start = time.time()
for _ in range(N):
    job_id = runner.execute_async([input_data], out_list)
    runner.wait(job_id)
elapsed = time.time() - start
print(f"\n{N} end-to-end inferences in {elapsed:.3f}s")
print(f"Avg latency: {elapsed/N*1000:.2f} ms")
print(f"Steady-state end-to-end FPS: {N/elapsed:.2f}")

# ---- Decode + NMS (matches your HLS IP math) ----
STRIDES = [8, 16, 32]
grid_x, grid_y, stride_arr = [], [], []
for s in STRIDES:
    g = 640 // s
    for yy in range(g):
        for xx in range(g):
            grid_x.append(xx); grid_y.append(yy); stride_arr.append(s)
grid_x = np.array(grid_x, dtype=np.float32)
grid_y = np.array(grid_y, dtype=np.float32)
stride_arr = np.array(stride_arr, dtype=np.float32)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def decode_and_nms(raw_output, conf_thresh=0.30, nms_thresh=0.45):
    raw = raw_output.reshape(-1, 7)
    cx = (raw[:, 0] + grid_x) * stride_arr
    cy = (raw[:, 1] + grid_y) * stride_arr
    w  = np.exp(raw[:, 2]) * stride_arr
    h  = np.exp(raw[:, 3]) * stride_arr
    obj = sigmoid(raw[:, 4])
    score_s = obj * sigmoid(raw[:, 5])
    score_f = obj * sigmoid(raw[:, 6])
    is_fire = score_f > score_s
    best = np.where(is_fire, score_f, score_s)
    cls  = np.where(is_fire, 1, 0)
    keep = np.where(best > conf_thresh)[0]
    print(f"Pre-NMS candidates: {len(keep)}")
    if len(keep) == 0:
        return []
    x1 = cx[keep] - w[keep]/2; y1 = cy[keep] - h[keep]/2
    x2 = cx[keep] + w[keep]/2; y2 = cy[keep] + h[keep]/2
    scores = best[keep]; classes = cls[keep]
    order = scores.argsort()[::-1]
    x1,y1,x2,y2 = x1[order],y1[order],x2[order],y2[order]
    scores,classes = scores[order],classes[order]
    keep_final = []
    suppressed = np.zeros(len(order), dtype=bool)
    for i in range(len(order)):
        if suppressed[i]: continue
        keep_final.append(i)
        if i == len(order)-1: break
        rest = np.arange(i+1, len(order))
        rest = rest[~suppressed[rest]]
        rest = rest[classes[rest] == classes[i]]
        if len(rest) == 0: continue
        xx1=np.maximum(x1[i],x1[rest]); yy1=np.maximum(y1[i],y1[rest])
        xx2=np.minimum(x2[i],x2[rest]); yy2=np.minimum(y2[i],y2[rest])
        inter=np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
        area_i=(x2[i]-x1[i])*(y2[i]-y1[i])
        area_r=(x2[rest]-x1[rest])*(y2[rest]-y1[rest])
        iou=inter/(area_i+area_r-inter)
        suppressed[rest[iou>nms_thresh]] = True
    return [(x1[i],y1[i],x2[i],y2[i],scores[i],classes[i]) for i in keep_final]

results = decode_and_nms(out_list[0])
names = ['smoke', 'fire']
print(f"\nPost-NMS detections: {len(results)}")
for x1,y1,x2,y2,score,cls in results:
    print(f"{names[int(cls)]}: score={score:.3f} box=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")
