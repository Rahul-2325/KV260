import json, os, numpy as np, collections
PLAN='/home/root/weights/layer_plan.json'; DATA='/home/root/weights/data'
plan=json.load(open(PLAN)); files=sorted(os.listdir(DATA))
stems={f[:-4]:f for f in files if f.endswith('.npy')}
def find(name):
    n=name[:-4] if name.endswith('_fix') else name
    c=[f for s,f in stems.items() if s.endswith(n)]
    if not c: c=[f for s,f in stems.items() if s.endswith(n.lstrip('_'))]
    return c[0] if len(c)==1 else None
# convention: input_0=bias(const), input_2=weight(const), input_1=activation
CONST_SLOTS=('input_0','input_2')
need=0; got=0; used=set(); missing=[]; bad_shape=[]
per_op=collections.Counter()
for L in plan:
    for k in CONST_SLOTS:
        nm=L.get(k+'_name'); sh=L.get(k+'_shape')
        if nm is None: continue
        need+=1
        r=find(nm)
        if r:
            got+=1; used.add(r)
            a=np.load(os.path.join(DATA,r))
            if list(a.shape)!=list(sh): bad_shape.append((L['id'],k,list(a.shape),sh))
        else:
            missing.append((L['id'],L['op_type'],k,nm,sh)); per_op[L['op_type']]+=1
print("const tensors referenced by plan : %d" % need)
print("  resolved to a file             : %d" % got)
print("  MISSING                        : %d" % len(missing))
print("  shape mismatches               : %d" % len(bad_shape))
print("  files used %d / %d  (unused: %d)" % (len(used), len(files), len(files)-len(used)))
print("\nmissing by op_type:", dict(per_op))
print("\nMISSING LIST:")
for m in missing: print("  L%-4d %-16s %-8s %-44s %s" % m[:1]+m[1:2]+m[2:3]+(m[3][:44],)+(m[4],) if False else "  L%-4d %-16s %-8s %-44s %s" % (m[0],m[1],m[2],m[3][:44],m[4]))
for b in bad_shape[:5]: print("  SHAPE MISMATCH L%d %s file=%s plan=%s" % b)
aff=sorted(set(m[0] for m in missing))
print("\naffected layers: %s  (%d of %d)" % (aff, len(aff), len(plan)))
