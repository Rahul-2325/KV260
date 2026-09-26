import xir, vart
import numpy as np
import cv2
import time

XMODEL = '/workspace/compiled_v5/lcam_v5.xmodel'
IMG_PATH = '/home/root/WEB09971.jpg'

graph = xir.Graph.deserialize(XMODEL)
root = graph.get_root_subgraph()
dpu_subgraphs = [sg for sg in root.toposort_child_subgraph()
                 if sg.has_attr('device') and sg.get_attr('device') == 'DPU']

runner = vart.Runner.create_runner(dpu_subgraphs[0], "run")
input_tensors = runner.get_input_tensors()
output_tensors = runner.get_output_tensors()

t_in = input_tensors[0]
print("Input tensor name:", t_in.name)
print("Input tensor dtype:", t_in.dtype)
print("Input tensor dims:", t_in.dims)

img0 = cv2.imread(IMG_PATH)
h0, w0 = img0.shape[:2]
r = min(640/h0, 640/w0)
rimg = cv2.resize(img0, (int(w0*r), int(h0*r)))
canvas = np.full((640, 640, 3), 114, np.uint8)
canvas[:rimg.shape[0], :rimg.shape[1]] = rimg

# Feed int8 directly - matches the tensor's actual dtype (xint8), no float anywhere
input_int8 = canvas.astype(np.int8)[np.newaxis, ...]

input_buffers  = [np.empty(tuple(t_in.dims), dtype=np.int8, order='C')]
output_buffers = [np.empty(tuple(t.dims), dtype=np.float32, order='C') for t in output_tensors]
np.copyto(input_buffers[0], input_int8)

print("Warming up (int8 input path)...")
for _ in range(10):
    job_id = runner.execute_async(input_buffers, output_buffers)
    runner.wait(job_id)

print("Timing (int8 input, single DPU subgraph)...")
t_total = 0.0
N = 100
for _ in range(N):
    t0 = time.perf_counter()
    job_id = runner.execute_async(input_buffers, output_buffers)
    runner.wait(job_id)
    t1 = time.perf_counter()
    t_total += (t1 - t0)

total_ms = t_total / N * 1000
print(f"\nTotal per-frame (int8 input): {total_ms:.3f} ms")
print(f"FPS: {1000/total_ms:.2f}")
