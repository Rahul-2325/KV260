import json, os, numpy as np
PLAN='/home/root/weights/layer_plan.json'; DATA='/home/root/weights/data'
plan=json.load(open(PLAN)); files=os.listdir(DATA)
stems={f[:-4]:f for f in files if f.endswith('.npy')}
def find(name):
    if not name: return None
    n=name[:-4] if name.endswith('_fix') else name
    cands=[f for s,f in stems.items() if s.endswith(n)]
    if not cands:
        n2=n.lstrip('_')
        cands=[f for s,f in stems.items() if s.endswith(n2)]
    return cands[0] if len(cands)==1 else (cands if cands else None)

need=0; got=0; amb=0; miss=[]
for L in plan:
    for k in ('input_0','input_1','input_2'):
        nm=L.get(k+'_name'); sh=L.get(k+'_shape')
        if nm is None: continue
        # heuristic: activation tensors have 4D shape starting with 1; consts don't
        is_act = isinstance(sh,list) and len(sh)==4 and sh[0]==1
        if is_act: continue
        need+=1
        r=find(nm)
        if isinstance(r,str): got+=1
        elif isinstance(r,list): amb+=1; miss.append((L['id'],k,nm,'AMBIG %d'%len(r)))
        else: miss.append((L['id'],k,nm,'MISSING'))
print("const tensors needed : %d" % need)
print("  matched uniquely   : %d" % got)
print("  ambiguous          : %d" % amb)
print("  missing            : %d" % (need-got-amb))
for m in miss[:12]: print("   ", m)
# shape check on a few
print("\nshape verification (first 6):")
c=0
for L in plan:
    for k in ('input_0','input_2'):
        nm=L.get(k+'_name'); sh=L.get(k+'_shape')
        if not nm or (isinstance(sh,list) and len(sh)==4 and sh[0]==1): continue
        r=find(nm)
        if isinstance(r,str):
            a=np.load(os.path.join(DATA,r))
            ok = list(a.shape)==list(sh)
            print("  L%-3d %-8s plan=%-16s npy=%-16s %s" % (L['id'],k,sh,list(a.shape),"OK" if ok else "MISMATCH"))
            c+=1
    if c>=6: break
