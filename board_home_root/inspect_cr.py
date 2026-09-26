import numpy as np, os
D='/home/root/weights/data'
for f in ('const_tial_attention_channel_reduce_avg_weight.npy',
          'const_tial_attention_channel_reduce_max_weight.npy'):
    a=np.load(os.path.join(D,f))
    u=np.unique(a)
    print("%s\n  shape=%s dtype=%s uniques=%d" % (f, list(a.shape), a.dtype, u.size))
    print("  first 16 vals: %s" % a.flatten()[:16].tolist())
    if u.size<=6: print("  UNIQUE VALUES: %s  counts=%s" % (u.tolist(), [int((a==v).sum()) for v in u]))
    print()
# also look at a spatial attention conv weight for context
f='const_bone_lcam5_spatial_attention_conv_weight.npy'
a=np.load(os.path.join(D,f))
print("%s shape=%s uniques=%d first=%s" % (f, list(a.shape), np.unique(a).size, a.flatten()[:12].tolist()))
