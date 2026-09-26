import json, time, numpy as np
from hw_ip_driver import ZoclBuffer, DRM_PATH_DEFAULT
from ip_wrappers import Conv2dEngine
plan=json.load(open('/home/root/weights/layer_plan.json'))
convs=[L for L in plan if L['op_type']=='conv2d-fix']
def macs(L):
    o=L['out_shape']; k=L['kernel']; ci=L['input_2_shape'][-1] if L.get('input_2_shape') else 1
    return o[1]*o[2]*o[3]*k[0]*k[1]*ci
convs.sort(key=macs)
print("conv layers: %d   MAC range: %.3fM .. %.1fM" % (len(convs), macs(convs[0])/1e6, macs(convs[-1])/1e6))
tot=sum(macs(L) for L in convs)
print("TOTAL conv MACs: %.2f G" % (tot/1e9))
BUF=8<<20
bi=ZoclBuffer(DRM_PATH_DEFAULT,BUF); bw=ZoclBuffer(DRM_PATH_DEFAULT,BUF)
bb=ZoclBuffer(DRM_PATH_DEFAULT,BUF); bo=ZoclBuffer(DRM_PATH_DEFAULT,BUF)
ip=Conv2dEngine()
print("\n%-6s %-20s %-10s %s" % ("id","out_shape","MACs","HW time"))
for L in [convs[0], convs[len(convs)//4], convs[len(convs)//2]]:
    o=L['out_shape']; k=L['kernel']; ci=L['input_2_shape'][-1]
    H,W,Co=o[1],o[2],o[3]
    try:
        t=ip.run(bi.phys_addr,bw.phys_addr,bb.phys_addr,bo.phys_addr,
                 H,W,ci,Co,k[0],k[1],L['stride'][0],L['stride'][1],
                 L['pad'][0],L['pad'][1],
                 L.get('input_1_fp') or 0, L.get('input_2_fp') or 0,
                 L.get('input_0_fp') or 0, L['fp_out'], 0)
        print("%-6d %-20s %-10.2fM %.2f ms" % (L['id'], str(o), macs(L)/1e6, t*1000))
    except Exception as e:
        print("%-6d %-20s %-10.2fM TIMEOUT/ERR: %s" % (L['id'], str(o), macs(L)/1e6, str(e)[:60]))
ip.close()
for b in (bi,bw,bb,bo): b.close()
