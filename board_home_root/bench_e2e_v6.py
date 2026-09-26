import xir, vart
import numpy as np
import cv2
import time
import threading
import queue

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

img0 = cv2.imread(IMG_PATH)
h0, w0 = img0.shape[:2]
r = min(640/h0, 640/w0)
rimg = cv2.resize(img0, (int(w0*r), int(h0*r)))
canvas = np.full((640, 640, 3), 114, np.uint8)
canvas[:rimg.shape[0], :rimg.shape[1]] = rimg
input_int8 = canvas.astype(np.int8)[np.newaxis, ...]

# ---- TEST 1: double-buffered overlap ----
# Two separate buffer sets so job N+1 can be dispatched into buffer B
# while job N (buffer A) is still finishing.
buf_a_in  = np.empty(tuple(t_in.dims), dtype=np.int8, order='C')
buf_b_in  = np.empty(tuple(t_in.dims), dtype=np.int8, order='C')
np.copyto(buf_a_in, input_int8)
np.copyto(buf_b_in, input_int8)
buf_a_out = [np.empty(tuple(t.dims), dtype=np.float32, order='C') for t in output_tensors]
buf_b_out = [np.empty(tuple(t.dims), dtype=np.float32, order='C') for t in output_tensors]

print("Warming up...")
for _ in range(5):
    jid = runner.execute_async([buf_a_in], buf_a_out)
    runner.wait(jid)

print("\nTEST 1: overlapped dispatch (queue job B before waiting on job A)")
N = 100
t0 = time.perf_counter()
jid_prev = runner.execute_async([buf_a_in], buf_a_out)
for i in range(N):
    use_a = (i % 2 == 0)
    next_buf_in  = buf_b_in if use_a else buf_a_in
    next_buf_out = buf_b_out if use_a else buf_a_out
    jid_next = runner.execute_async([next_buf_in], next_buf_out)
    runner.wait(jid_prev)
    jid_prev = jid_next
runner.wait(jid_prev)
t1 = time.perf_counter()
overlap_ms = (t1 - t0) / N * 1000
print(f"Overlapped: {overlap_ms:.3f} ms/frame -> {1000/overlap_ms:.2f} FPS")

# ---- TEST 2: serial baseline for direct comparison (same run, same conditions) ----
print("\nTEST 2: serial baseline (same buffers, same loop, no overlap)")
t0 = time.perf_counter()
for i in range(N):
    jid = runner.execute_async([buf_a_in], buf_a_out)
    runner.wait(jid)
t1 = time.perf_counter()
serial_ms = (t1 - t0) / N * 1000
print(f"Serial: {serial_ms:.3f} ms/frame -> {1000/serial_ms:.2f} FPS")

print(f"\nSpeedup from overlap: {serial_ms/overlap_ms:.2f}x")
