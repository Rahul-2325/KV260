from hw_ip_driver import ZoclBuffer, DRM_PATH_DEFAULT
import numpy as np
in_buf = ZoclBuffer(DRM_PATH_DEFAULT, 48)
out_buf = ZoclBuffer(DRM_PATH_DEFAULT, 48)
print("buffers created OK, phys addrs:", hex(in_buf.phys_addr), hex(out_buf.phys_addr))
in_buf.write_array(np.arange(48, dtype=np.int8))
in_buf.sync_to_device()
print("write+sync OK - NO IP CALL MADE")
in_buf.close(); out_buf.close()
print("DONE")