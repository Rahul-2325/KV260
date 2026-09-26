from hw_ip_driver import ZoclBuffer, DRM_PATH_DEFAULT
from ip_wrappers import PoolEngine
import numpy as np
H, W, C = 4, 4, 2
BIG = 1024*1024
in_buf  = ZoclBuffer(DRM_PATH_DEFAULT, BIG)
out_buf = ZoclBuffer(DRM_PATH_DEFAULT, BIG)
print("in=0x%x out=0x%x" % (in_buf.phys_addr, out_buf.phys_addr))
src = np.zeros((H, W, C), dtype=np.int8); src[:,:,0] = 10; src[:,:,1] = 20
in_buf.write_array(src); in_buf.sync_to_device()
out_buf.write_array(np.full(C, 0x55, dtype=np.int8)); out_buf.sync_to_device()
ip = PoolEngine()
el = ip.run(in_buf.phys_addr, out_buf.phys_addr, H, W, C, 0, 0)
ip.close()
out_buf.sync_from_device()
res = out_buf.read_array((C,), np.int8).tolist()
print("HW call %.4f ms" % (el*1000))
print("pool_engine (smartconnect_hp1) result: %s   expected approx [10, 20]" % res)
print("*** PASS ***" if res == [10,20] else ("SENTINEL INTACT - no write" if res==[0x55,0x55] else "value: %s" % res))
in_buf.close(); out_buf.close()
