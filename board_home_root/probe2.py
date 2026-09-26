import xir
g = xir.Graph.deserialize('/home/root/lcam_v5.xmodel')
dpu = [s for s in g.get_root_subgraph().toposort_child_subgraph()
       if s.has_attr("device") and s.get_attr("device")=="DPU"]
s = dpu[0]
a = s.get_attrs()
print("get_attrs type:", type(a).__name__)
for k in ("reg_id_to_parameter_value","reg_id_to_size","reg_id_to_context_type","mc_code"):
    if k in a:
        v = a[k]
        print("%-28s type=%-12s %s" % (k, type(v).__name__,
              (str({kk: (type(vv).__name__, len(bytes(vv)) if isinstance(vv,(bytes,bytearray)) else vv)
                    for kk,vv in v.items()})[:200]) if isinstance(v,dict)
              else (len(bytes(v)) if isinstance(v,(bytes,bytearray)) else str(v)[:120])))
    else:
        print("%-28s ABSENT from get_attrs()" % k)
