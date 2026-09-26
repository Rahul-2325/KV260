
import numpy as np, xir, vart
g = xir.Graph.deserialize("/home/root/lcam_hybrid.xmodel")
root = g.get_root_subgraph()
subs = [s for s in root.toposort_child_subgraph()
        if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]
r = vart.Runner.create_runner(subs[0], "run")
it, ot = r.get_input_tensors(), r.get_output_tensors()
ins  = [np.zeros(tuple(t.dims), dtype=np.int8) for t in it]
outs = [np.zeros(tuple(t.dims), dtype=np.int8) for t in ot]
print("child: submitting", flush=True)
r.wait(r.execute_async(ins, outs))
print("child: COMPLETED", flush=True)
