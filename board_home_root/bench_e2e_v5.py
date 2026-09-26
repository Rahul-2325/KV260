import xir, vart
import numpy as np
import time

XMODEL = '/workspace/compiled_v5/lcam_v5.xmodel'

graph = xir.Graph.deserialize(XMODEL)
root = graph.get_root_subgraph()
dpu_subgraphs = [sg for sg in root.toposort_child_subgraph()
                 if sg.has_attr('device') and sg.get_attr('device') == 'DPU']

runner = vart.Runner.create_runner(dpu_subgraphs[0], "run")
input_tensors = runner.get_input_tensors()
output_tensors = runner.get_output_tensors()
t_in = input_tensors[0]

input_buffers  = [np.zeros(tuple(t_in.dims), dtype=np.int8, order='C')]
output_buffers = [np.empty(tuple(t.dims), dtype=np.float32, order='C') for t in output_tensors]

print("Per-call timing, first 20 calls individually:")
for i in range(20):
    t0 = time.perf_counter()
    job_id = runner.execute_async(input_buffers, output_buffers)
    runner.wait(job_id)
    t1 = time.perf_counter()
    print(f"  call {i:2d}: {(t1-t0)*1000:.3f} ms")

print("\nNow 100 more, just min/avg/max:")
times = []
for i in range(100):
    t0 = time.perf_counter()
    job_id = runner.execute_async(input_buffers, output_buffers)
    runner.wait(job_id)
    t1 = time.perf_counter()
    times.append((t1-t0)*1000)
times = np.array(times)
print(f"min: {times.min():.3f} ms, avg: {times.mean():.3f} ms, max: {times.max():.3f} ms")
