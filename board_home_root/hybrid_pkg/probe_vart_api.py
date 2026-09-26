#!/usr/bin/env python3
"""
probe_vart_api.py -- run ON THE BOARD.

Can we hand the LCAM CU a buffer that VART will use directly, instead of
copying the CU's output out through a write-combining mapping at 135 MB/s?

That read is now the pipeline's largest single cost (21.5 ms of a 94 ms
frame). Allocation flags do not fix it -- CMA, CMA|COHERENT and COHERENT
all read at ~135 MB/s (bench_bo_flags.py). The only real fix is to stop
doing the copy: point the CU's feat_out at the physical address of the
buffer the NEXT DPU subgraph reads from.

For that we need, from the Python binding:
  * a TensorBuffer object exposed to Python
  * its physical/device address
  * Runner.execute_async accepting TensorBuffers, not just numpy arrays

This prints what the installed vart/xir modules actually expose, so the
question is settled from the API rather than by assumption.
"""
import numpy as np
import vart
import xir

print("=== vart module ===")
print([n for n in dir(vart) if not n.startswith("_")])

print("\n=== vart.Runner ===")
print([n for n in dir(vart.Runner) if not n.startswith("_")])

for cls in ("TensorBuffer", "RunnerExt"):
    if hasattr(vart, cls):
        print("\n=== vart.%s ===" % cls)
        print([n for n in dir(getattr(vart, cls)) if not n.startswith("_")])
    else:
        print("\n=== vart.%s : NOT PRESENT ===" % cls)

g = xir.Graph.deserialize("/home/root/lcam_hybrid.xmodel")
subs = [s for s in g.get_root_subgraph().toposort_child_subgraph()
        if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]
r = vart.Runner.create_runner(subs[0], "run")
print("\n=== instance of Runner: %s ===" % type(r))
print([n for n in dir(r) if not n.startswith("_")])

# RunnerExt exposes get_inputs()/get_outputs() returning TensorBuffers with
# device addresses on some builds -- that is exactly what we would need.
if hasattr(r, "get_inputs"):
    tbs = r.get_inputs()
    print("\nrunner.get_inputs() -> %d TensorBuffer(s)" % len(tbs))
    tb = tbs[0]
    print("  type: %s" % type(tb))
    print("  attrs: %s" % [n for n in dir(tb) if not n.startswith("_")])
    for meth in ("data", "data_phy"):
        if hasattr(tb, meth):
            try:
                print("  %s() -> %s" % (meth, getattr(tb, meth)([0, 0, 0, 0])))
            except Exception as e:
                try:
                    print("  %s() -> %s" % (meth, getattr(tb, meth)()))
                except Exception as e2:
                    print("  %s() raised %s / %s" % (meth, e, e2))
else:
    print("\nrunner has NO get_inputs() -- numpy-only interface, so the CU"
          "\noutput must be copied through the CPU unless we switch to"
          "\nRunnerExt or the C++ API.")
