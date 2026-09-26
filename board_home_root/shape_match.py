import json, os, numpy as np
PLAN='/home/root/weights/layer_plan.json'; DATA='/home/root/weights/data'
plan=json.load(open(PLAN)); files=sorted(os.listdir(DATA))
stems={f[:-4]:f for f in files if f.endswith('.npy')}
def find(name):
    n=name[:-4] if name.endswith('_fix') else name
    c=[f for s,f in stems.items() if s.endswith(n)]
    if not c:
        c=[f for s,f in stems.items() if s.endswith(n.lstrip('_'))]
    return c[0] if len(c)==1 else None
used=set(); missing=[]
for L in plan:
    for k in ('input_0','input_1','input_2'):
        nm=L.get(k+'_name'); sh=L.get(k+'_shape')
        if nm is None: continue
        if isinstance(sh,list) and len(sh)==4 and sh[0]==1: continue
        r=find(nm)
        if r: used.add(r)
        else: missing.append((L['id'],k,nm,sh))
unused=[f for f in files if f not in used]
print("used: %d   UNUSED: %d" % (len(used), len(unused)))
print("\nUNUSED FILES (shape):")
for f in unused:
    a=np.load(os.path.join(DATA,f))
    print("  %-62s %s" % (f[:62], list(a.shape)))
print("\nMISSING TENSORS (wanted shape) -> shape-matched candidates:")
for lid,k,nm,sh in missing:
    cands=[f for f in unused if list(np.load(os.path.join(DATA,f)).shape)==list(sh)]
    print("  L%-4d %-8s %-44s want=%-16s" % (lid,k,nm[:44],sh))
    for c in cands: print("        CANDIDATE: %s" % c)
    if not cands: print("        (no unused file with that shape)")
