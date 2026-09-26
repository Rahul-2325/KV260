import numpy as np
z = np.load('/home/root/e2e_outputs.npz', allow_pickle=True)
p = z['out_0']
while p.ndim > 2: p = p[0]
print("shape", p.shape, p.dtype)
print("global: min=%.4f max=%.4f mean=%.4f std=%.4f" % (p.min(), p.max(), p.mean(), p.std()))
names = ['cx','cy','w','h','obj','cls0','cls1']
print("\n%-5s %10s %10s %10s %10s %8s" % ("chan","min","max","mean","std","uniq"))
for i in range(p.shape[1]):
    c = p[:, i]
    print("%-5s %10.4f %10.4f %10.4f %10.4f %8d" % (names[i], c.min(), c.max(), c.mean(), c.std(), len(np.unique(c))))
print("\nfirst 5 anchors:")
for r in range(5):
    print("  ", np.round(p[r], 4).tolist())
print("\nanchors 4000-4004:")
for r in range(4000, 4005):
    print("  ", np.round(p[r], 4).tolist())