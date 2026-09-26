import json
plan=json.load(open('/home/root/weights/layer_plan.json'))
def macs(L):
    o=L['out_shape']; k=L.get('kernel') or [1,1]
    ci=L['input_2_shape'][-1] if L.get('input_2_shape') else 1
    return o[1]*o[2]*o[3]*k[0]*k[1]*ci
def elems(L):
    o=L['out_shape']; return o[1]*o[2]*o[3]
conv=[L for L in plan if L['op_type']=='conv2d-fix']
other=[L for L in plan if L['op_type']!='conv2d-fix']
G=sum(macs(L) for L in conv)
# measured points
p=[(0.63e6,0.20749),(52.43e6,3.32895)]
# linear fit: t = a*MACs + b  (two points)
a=(p[1][1]-p[0][1])/(p[1][0]-p[0][0]); b=p[0][1]-a*p[0][0]
print("measured throughput: %.2f MMAC/s (slope)" % (1/a/1e6))
print("per-layer fixed overhead: %.1f ms" % (b*1000))
est_conv=sum(a*macs(L)+b for L in conv)
est_other=sum(b + elems(L)*a*1.0 for L in other)   # rough: elementwise ~1 op each
print("\nconv layers   : %3d   %.2f GMAC   est %.1f s" % (len(conv), G/1e9, est_conv))
print("other layers  : %3d               est %.1f s" % (len(other), est_other))
tot=est_conv+est_other
print("\nESTIMATED TOTAL PER FRAME : %.1f s  (%.1f min)" % (tot, tot/60))
print("ESTIMATED FPS             : %.5f" % (1/tot))
print("DPU baseline              : 195 FPS")
print("=> custom accelerator is  ~%.0fx SLOWER than DPU" % (195*tot))
print("\ntop-10 heaviest conv layers:")
for L in sorted(conv,key=macs,reverse=True)[:10]:
    print("   L%-4d %-22s %7.1f MMAC  est %6.1f s" % (L['id'],str(L['out_shape']),macs(L)/1e6,a*macs(L)+b))
