import numpy as np, os
D='/home/root/weights/data'
fs=sorted(os.listdir(D))
tot=0
for f in fs[:6]:
    a=np.load(os.path.join(D,f))
    tot+=a.nbytes
    print("%-58s shape=%-16s dtype=%-8s min=%-8s max=%-8s" % (f[:58], list(a.shape), a.dtype, a.min(), a.max()))
print("...")
allb=sum(np.load(os.path.join(D,f)).nbytes for f in fs)
dts={}
for f in fs:
    d=str(np.load(os.path.join(D,f)).dtype); dts[d]=dts.get(d,0)+1
print("\ndtypes across all 211:", dts)
print("total bytes: %.2f MB" % (allb/1e6))
