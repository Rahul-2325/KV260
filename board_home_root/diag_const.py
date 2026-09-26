import xir, numpy as np
g = xir.Graph.deserialize('/home/root/lcam_v5.xmodel')
consts = [o for o in g.get_ops() if 'const' in o.get_type()]
empty = full = 0
sample_full = sample_empty = None
for o in consts:
    raw = o.get_attr('data')
    n = len(bytes(raw))
    if n == 0:
        empty += 1
        if sample_empty is None: sample_empty = o
    else:
        full += 1
        if sample_full is None: sample_full = o
print("const ops: %d   with data: %d   EMPTY: %d" % (len(consts), full, empty))
if sample_full is not None:
    o = sample_full; t = o.get_output_tensor()
    raw = o.get_attr('data')
    print("\nGOOD example:", o.get_name()[-50:])
    print("  type(raw)=%s len=%d dims=%s dtype=%s" % (type(raw).__name__, len(bytes(raw)), list(t.dims), t.dtype))
if sample_empty is not None:
    o = sample_empty; t = o.get_output_tensor()
    raw = o.get_attr('data')
    print("\nEMPTY example:", o.get_name()[-50:])
    print("  type(raw)=%s len=%d dims=%s dtype=%s" % (type(raw).__name__, len(bytes(raw)), list(t.dims), t.dtype))
    print("  op attrs:", list(o.get_attrs().keys()))
    print("  tensor attrs:", list(t.get_attrs().keys()) if hasattr(t,'get_attrs') else 'n/a')
    print("  fanout ops:", [f.get_type() for f in o.get_fanout_ops()][:5])
