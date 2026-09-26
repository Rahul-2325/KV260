import xir, numpy as np
g = xir.Graph.deserialize('/home/root/lcam_v5.xmodel')
subs = g.get_root_subgraph().toposort_child_subgraph()
print("child subgraphs:", len(subs))
dpu = [s for s in subs if s.has_attr("device") and s.get_attr("device")=="DPU"]
print("DPU subgraphs:", len(dpu))
s = dpu[0]
print("\nsubgraph:", s.get_name()[:60])
print("attrs:", list(s.get_attrs().keys()))
for key in ("reg_id_to_parameter_value","reg_id_to_size","reg_id_to_context_type"):
    if s.has_attr(key):
        v = s.get_attr(key)
        if key == "reg_id_to_parameter_value":
            print("  %s: %s" % (key, {k: len(bytes(vv)) for k,vv in v.items()}))
        else:
            print("  %s: %s" % (key, v))
# total param bytes across all DPU subgraphs
tot = 0
for s in dpu:
    if s.has_attr("reg_id_to_parameter_value"):
        for k,vv in s.get_attr("reg_id_to_parameter_value").items():
            tot += len(bytes(vv))
print("\nTOTAL parameter bytes across all DPU subgraphs: %d (%.2f MB)" % (tot, tot/1e6))
