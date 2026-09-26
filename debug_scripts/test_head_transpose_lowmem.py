from hw_ip_driver import ZoclBuffer, DRM_PATH_DEFAULT
from ip_wrappers import HeadTranspose
import numpy as np

H, W, C = 4, 4, 3
total = H * W * C
BIG = 1024 * 1024          # 1MB -> forces CMA (low DDR) allocation

src = np.arange(total, dtype=np.int8)

in_buf  = ZoclBuffer(DRM_PATH_DEFAULT, BIG)
out_buf = ZoclBuffer(DRM_PATH_DEFAULT, BIG)
print("in_buf.phys_addr  = 0x%x" % in_buf.phys_addr)
print("out_buf.phys_addr = 0x%x" % out_buf.phys_addr)
assert in_buf.phys_addr < 0x80000000, "in_buf STILL HIGH - abort"
assert out_buf.phys_addr < 0x80000000, "out_buf STILL HIGH - abort"
print("both buffers are in LOW DDR (PL-reachable)")

in_buf.write_array(src)
in_buf.sync_to_device()

# sentinel so we can tell "wrote zeros" from "never wrote"
out_buf.write_array(np.full(total, 0x55, dtype=np.int8))
out_buf.sync_to_device()

ip = HeadTranspose()
elapsed = ip.run(in_buf.phys_addr, out_buf.phys_addr, H, W, C)
ip.close()
print("HW call completed in %.4f ms" % (elapsed*1000))

out_buf.sync_from_device()
result = out_buf.read_array((H, W, C), np.int8)
got = result.flatten().tolist()
exp = [0,16,32,1,17,33,2,18,34,3,19,35,4,20,36,5,21,37,6,22,38,7,23,39,
       8,24,40,9,25,41,10,26,42,11,27,43,12,28,44,13,29,45,14,30,46,15,31,47]
print("HW result: %s" % got)
print("Expected:  %s" % exp)
if got == exp:
    print("*** PASS -- HARDWARE PRODUCED CORRECT OUTPUT ***")
elif all(v == 0x55 for v in got):
    print("FAIL: sentinel intact -- PL never wrote")
elif all(v == 0 for v in got):
    print("FAIL: all zeros")
else:
    print("PARTIAL/DIFFERENT -- see above")
in_buf.close(); out_buf.close()
