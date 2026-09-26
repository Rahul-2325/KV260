#!/usr/bin/env python3
"""
probe_runtime.py
Determines what end-to-end inference support exists on the board:
  - subgraph breakdown of lcam_v5.xmodel (DPU vs CPU vs USER)
  - whether vart.Runner / GraphRunner are available
  - whether opencv is present for pre/post-processing
Run this BEFORE writing the end-to-end pipeline so we build against what
actually exists rather than assuming.
"""
import sys

print("=" * 64)
print("1. PYTHON / LIB AVAILABILITY")
print("=" * 64)
for mod in ("numpy", "cv2", "xir", "vart"):
    try:
        m = __import__(mod)
        v = getattr(m, "__version__", "?")
        print("  %-8s OK   version=%s" % (mod, v))
    except Exception as e:
        print("  %-8s MISSING (%s)" % (mod, e))

try:
    import xir
except Exception:
    print("\nxir unavailable - cannot continue")
    sys.exit(1)

print()
print("=" * 64)
print("2. GRAPH RUNNER AVAILABILITY (handles mixed CPU+DPU automatically)")
print("=" * 64)
gr = None
for path in ("vart.RunnerExt", "xir.GraphRunner", "vitis_ai_library.GraphRunner"):
    try:
        mod_name, attr = path.rsplit(".", 1)
        mod = __import__(mod_name, fromlist=[attr])
        gr = getattr(mod, attr)
        print("  FOUND: %s" % path)
    except Exception as e:
        print("  no    %s  (%s)" % (path, str(e)[:50]))
try:
    import graph_runner
    print("  FOUND: graph_runner module")
except Exception as e:
    print("  no    graph_runner  (%s)" % str(e)[:50])

print()
print("=" * 64)
print("3. XMODEL SUBGRAPH BREAKDOWN")
print("=" * 64)
g = xir.Graph.deserialize('/home/root/lcam_v5.xmodel')
subs = g.get_root_subgraph().toposort_child_subgraph()
print("  total child subgraphs: %d" % len(subs))
counts = {}
for i, s in enumerate(subs):
    dev = s.get_attr("device") if s.has_attr("device") else "NONE"
    counts[dev] = counts.get(dev, 0) + 1
print("  by device: %s" % counts)
print()
print("  %-4s %-8s %-7s %s" % ("idx", "device", "#ops", "name"))
for i, s in enumerate(subs):
    dev = s.get_attr("device") if s.has_attr("device") else "NONE"
    n = len(s.get_children()) if s.get_children() else 0
    print("  %-4d %-8s %-7d %s" % (i, dev, n, s.get_name()[:58]))

print()
print("=" * 64)
print("4. CPU SUBGRAPH DETAIL (these are the LCAM attention parts)")
print("=" * 64)
for i, s in enumerate(subs):
    dev = s.get_attr("device") if s.has_attr("device") else "NONE"
    if dev != "CPU":
        continue
    ops = s.get_children() or []
    print("  [%d] %s" % (i, s.get_name()[:60]))
    for o in ops:
        try:
            t = o.get_output_tensor()
            print("        %-18s out=%s" % (o.get_type(), list(t.dims)))
        except Exception:
            print("        %-18s" % o.get_type())

print()
print("=" * 64)
print("5. MODEL INPUT / OUTPUT TENSORS")
print("=" * 64)
root = g.get_root_subgraph()
try:
    for t in g.get_input_tensors():
        print("  INPUT : %-40s %s" % (t.name[:40], list(t.dims)))
except Exception as e:
    print("  (graph-level input query failed: %s)" % e)
try:
    for t in g.get_output_tensors():
        print("  OUTPUT: %-40s %s" % (t.name[:40], list(t.dims)))
except Exception as e:
    print("  (graph-level output query failed: %s)" % e)
