import json, collections
plan=json.load(open('/home/root/weights/layer_plan.json'))
names={}
for L in plan: names[L['name']]=L['id']
# activation slots by op type
ACT={'conv2d-fix':['input_1'],'depthwise-fix':['input_0','input_1'],
     'pool-fix':['input_0'],'eltwise-fix':['input_0','input_1'],
     'hard-sigmoid-fix':['input_0']}
def producer(nm):
    if nm is None: return None
    hits=[i for n,i in names.items() if n.endswith(nm) or nm.endswith(n)]
    return hits[0] if len(hits)==1 else (hits if hits else None)
tot=res=amb=ext=0
externals=[]
for L in plan:
    for s in ACT.get(L['op_type'],[]):
        nm=L.get(s+'_name')
        if nm is None: continue
        tot+=1
        n2=nm[:-4] if nm.endswith('_fix') else nm
        p=producer(n2)
        if isinstance(p,int): res+=1
        elif isinstance(p,list): amb+=1
        else:
            ext+=1
            if len(externals)<12: externals.append((L['id'],L['op_type'],s,nm))
print("activation inputs total : %d" % tot)
print("  resolved to a layer   : %d" % res)
print("  ambiguous             : %d" % amb)
print("  NO producer (external): %d" % ext)
print("\nexternal/unresolved inputs:")
for e in externals: print("   L%-4d %-18s %-8s %s" % e)
print("\nlayer id order sane?", plan[0]['id'], "->", plan[-1]['id'], " ids sorted:", all(plan[i]['id']<=plan[i+1]['id'] for i in range(len(plan)-1)))
