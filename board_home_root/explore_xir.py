import xir
g = xir.Graph.deserialize('/home/root/lcam_v5.xmodel')
ops = g.get_ops()
print("total ops:", len(ops))
import collections
print("op types:", collections.Counter(o.get_type() for o in ops).most_common(12))
# find a const op
consts = [o for o in ops if 'const' in o.get_type()]
print("\nconst ops:", len(consts))
o = consts[0]
print("sample const name:", o.get_name())
print("Op methods:", [m for m in dir(o) if not m.startswith('_')])
t = o.get_output_tensor()
print("\nTensor methods:", [m for m in dir(t) if not m.startswith('_')])
print("tensor dims:", t.dims, "dtype:", t.dtype)
print("attrs:", o.get_attrs().keys() if hasattr(o,'get_attrs') else 'n/a')
