import xir, vitis_ai_library
import numpy as np
import cv2
import time

XMODEL = '/workspace/compiled_v5/lcam_v5.xmodel'
IMG_PATH = '/home/root/WEB09971.jpg'

graph = xir.Graph.deserialize(XMODEL)
runner = vitis_ai_library.GraphRunner.create_graph_runner(graph)
input_tensors = runner.get_input_tensors()
output_tensors = runner.get_output_tensors()
print("Input shape:", tuple(input_tensors[0].dims))

img0 = cv2.imread(IMG_PATH)
img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
h0, w0 = img0.shape[:2]
r = min(640/h0, 640/w0)
rimg = cv2.resize(img0, (int(w0*r), int(h0*r)))
canvas = np.full((640, 640, 3), 114, np.uint8)
canvas[:rimg.shape[0], :rimg.shape[1]] = rimg
input_data = canvas.astype(np.float32)[np.newaxis, ...]

out_list = [np.empty(tuple(t.dims), dtype=np.float32) for t in output_tensors]

print("Warming up...")
for _ in range(10):
    job_id = runner.execute_async([input_data], out_list)
    runner.wait(job_id)

print("Timing (phase-split, full 24-subgraph pipeline)...")
t_dispatch = 0.0
t_wait = 0.0
N = 100
for _ in range(N):
    t0 = time.perf_counter()
    job_id = runner.execute_async([input_data], out_list)
    t1 = time.perf_counter()
    runner.wait(job_id)
    t2 = time.perf_counter()
    t_dispatch += (t1 - t0)
    t_wait     += (t2 - t1)

total_ms = (t_dispatch + t_wait) / N * 1000
print(f"\nexecute_async avg: {t_dispatch/N*1000:.3f} ms")
print(f"wait avg:          {t_wait/N*1000:.3f} ms")
print(f"Total per-frame:   {total_ms:.3f} ms")
print(f"FPS:               {1000/total_ms:.2f}")
