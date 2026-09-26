import time, numpy as np
from hw_ip_driver import ZoclBuffer, DRM_PATH_DEFAULT
from ip_wrappers import LcamAttentionGate
BUF=8<<20
bi=ZoclBuffer(DRM_PATH_DEFAULT,BUF); bw=ZoclBuffer(DRM_PATH_DEFAULT,BUF); bo=ZoclBuffer(DRM_PATH_DEFAULT,BUF)
ip=LcamAttentionGate()
print("LCAM attention gate - real network sizes (the 4 gate layers)")
print("%-6s %-20s %-12s %-12s %s" % ("layer","shape","HW time","CPU numpy","winner"))
for lid,(H,W,C) in [(23,(160,160,64)),(41,(80,80,128)),(53,(40,40,256)),(83,(20,20,512))]:
    feat=np.random.randint(-128,127,(H,W,C),dtype=np.int8)
    gate=np.random.randint(-128,127,(C,),dtype=np.int8)
    bi.write_array(feat); bi.sync_to_device()
    bw.write_array(gate); bw.sync_to_device()
    try:
        t=ip.run(bi.phys_addr,bw.phys_addr,bo.phys_addr,H,W,C,4,7,4)
        hw="%.1f ms"%(t*1000)
    except Exception as e:
        hw="ERR/TO"; t=None
    t0=time.perf_counter()
    _=((feat.astype(np.int32)*gate.astype(np.int32))>>7).clip(-128,127).astype(np.int8)
    cpu=time.perf_counter()-t0
    win = "HW" if (t is not None and t<cpu) else "CPU"
    print("%-6d %-20s %-12s %-12s %s" % (lid,"%dx%dx%d"%(H,W,C),hw,"%.1f ms"%(cpu*1000),win))
ip.close()
for b in (bi,bw,bo): b.close()
