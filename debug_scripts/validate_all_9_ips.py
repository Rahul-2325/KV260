#!/usr/bin/env python3
"""
validate_all_9_ips.py
Exercises all 9 custom HLS IPs on real hardware and checks their output.

Requires the FIXED hw_ip_driver.py (MIN_CMA_ALLOC forcing low-DDR /
PL-reachable buffers -- see PROJECT_HISTORY.md §26). Before that fix every
one of these returned all zeros.

Each buffer is pre-filled with a 0x55 sentinel so "never written" is
distinguishable from "wrote zeros".
"""
import numpy as np
from hw_ip_driver import ZoclBuffer, DRM_PATH_DEFAULT
from ip_wrappers import (Conv2dEngine, DepthwiseEngine, EltwiseAdd, PoolEngine,
                         UpsampleEngine, LcamAttentionGate, HeadSigmoid,
                         HeadTranspose, HeadConcatReshape)

BIG = 1 << 20
SENT = 0x55
results = []

def buf(arr=None, fill=None):
    b = ZoclBuffer(DRM_PATH_DEFAULT, BIG)
    if arr is not None:
        b.write_array(np.ascontiguousarray(arr).astype(np.int8))
    elif fill is not None:
        b.write_array(np.full(fill, SENT, dtype=np.int8))
    b.sync_to_device()
    return b

def report(name, got, exp=None):
    got = list(got)
    if all(v == SENT for v in got):
        verdict, ok = "NEVER WRITTEN (sentinel intact)", False
    elif exp is not None:
        ok = got == list(exp)
        verdict = "PASS (exact match)" if ok else "MISMATCH"
    else:
        ok = True
        verdict = "wrote data (no exact reference)"
    results.append((name, ok))
    print("  %-22s %s" % (name, verdict))
    if exp is not None and got != list(exp):
        print("      got: %s" % got[:24])
        print("      exp: %s" % list(exp)[:24])
    elif exp is None:
        print("      out: %s" % got[:16])

print("=" * 62)
print("VALIDATING ALL 9 CUSTOM IPs ON HARDWARE")
print("=" * 62)

# ---- 1. head_transpose : NCHW -> NHWC -------------------------------------
H, W, C = 4, 4, 3
n = H * W * C
src = np.arange(n, dtype=np.int8)
i_b, o_b = buf(src), buf(fill=n)
ip = HeadTranspose(); ip.run(i_b.phys_addr, o_b.phys_addr, H, W, C); ip.close()
o_b.sync_from_device()
exp = [int(src[(c * H + h) * W + w]) for h in range(H) for w in range(W) for c in range(C)]
report("head_transpose", o_b.read_array((n,), np.int8), exp)
i_b.close(); o_b.close()

# ---- 2. pool_engine : global average pool over HxW ------------------------
H, W, C = 4, 4, 2
src = np.zeros((H, W, C), dtype=np.int8); src[:, :, 0] = 10; src[:, :, 1] = 20
i_b, o_b = buf(src), buf(fill=C)
ip = PoolEngine(); ip.run(i_b.phys_addr, o_b.phys_addr, H, W, C, 0, 0); ip.close()
o_b.sync_from_device()
report("pool_engine", o_b.read_array((C,), np.int8), [10, 20])
i_b.close(); o_b.close()

# ---- 3. upsample_engine : nearest-neighbour 2x ----------------------------
H, W, C, S = 2, 2, 2, 2
src = np.arange(H * W * C, dtype=np.int8).reshape(H, W, C)
n_out = (H * S) * (W * S) * C
i_b, o_b = buf(src), buf(fill=n_out)
ip = UpsampleEngine(); ip.run(i_b.phys_addr, o_b.phys_addr, H, W, C, S); ip.close()
o_b.sync_from_device()
exp = [int(src[h // S, w // S, c]) for h in range(H * S) for w in range(W * S) for c in range(C)]
report("upsample_engine", o_b.read_array((n_out,), np.int8), exp)
i_b.close(); o_b.close()

# ---- 4. eltwise_add : a + b (fp all 0 -> plain saturating add) ------------
H, W, C = 2, 2, 2
n = H * W * C
a = np.arange(n, dtype=np.int8)
b = np.full(n, 5, dtype=np.int8)
a_b, b_b, o_b = buf(a), buf(b), buf(fill=n)
ip = EltwiseAdd(); ip.run(a_b.phys_addr, b_b.phys_addr, o_b.phys_addr, H, W, C, 0, 0, 0); ip.close()
o_b.sync_from_device()
report("eltwise_add", o_b.read_array((n,), np.int8), [int(x) + 5 for x in a])
a_b.close(); b_b.close(); o_b.close()

# ---- 5. conv2d_engine : 1x1 kernel, weight=1, bias=0 -> identity ----------
H, W, Ci, Co = 3, 3, 1, 1
n = H * W * Ci
src = np.arange(1, n + 1, dtype=np.int8)
i_b = buf(src); w_b = buf(np.array([1], dtype=np.int8)); bi_b = buf(np.array([0], dtype=np.int8))
o_b = buf(fill=n)
ip = Conv2dEngine()
ip.run(i_b.phys_addr, w_b.phys_addr, bi_b.phys_addr, o_b.phys_addr,
       H, W, Ci, Co, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
ip.close(); o_b.sync_from_device()
report("conv2d_engine", o_b.read_array((n,), np.int8), [int(x) for x in src])
i_b.close(); w_b.close(); bi_b.close(); o_b.close()

# ---- 6. depthwise_engine : 1x1 kernel, weight=1, bias=0 -> identity -------
H, W, C = 3, 3, 2
n = H * W * C
src = np.arange(1, n + 1, dtype=np.int8)
i_b = buf(src); w_b = buf(np.ones(C, dtype=np.int8)); bi_b = buf(np.zeros(C, dtype=np.int8))
o_b = buf(fill=n)
ip = DepthwiseEngine()
ip.run(i_b.phys_addr, w_b.phys_addr, bi_b.phys_addr, o_b.phys_addr,
       H, W, C, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0)
ip.close(); o_b.sync_from_device()
report("depthwise_engine", o_b.read_array((n,), np.int8), [int(x) for x in src])
i_b.close(); w_b.close(); bi_b.close(); o_b.close()

# ---- 7. head_sigmoid ------------------------------------------------------
H, W, C = 2, 2, 2
n = H * W * C
src = np.array([-64, -32, 0, 32, 64, 96, -96, 16], dtype=np.int8)
i_b, o_b = buf(src), buf(fill=n)
ip = HeadSigmoid(); ip.run(i_b.phys_addr, o_b.phys_addr, H, W, C, 4, 7); ip.close()
o_b.sync_from_device()
report("head_sigmoid", o_b.read_array((n,), np.int8))
i_b.close(); o_b.close()

# ---- 8. lcam_attention_gate ----------------------------------------------
H, W, C = 2, 2, 2
n = H * W * C
feat = np.arange(1, n + 1, dtype=np.int8)
wgt = np.full(H * W, 64, dtype=np.int8)
f_b, w_b, o_b = buf(feat), buf(wgt), buf(fill=n)
ip = LcamAttentionGate()
ip.run(f_b.phys_addr, w_b.phys_addr, o_b.phys_addr, H, W, C, 4, 7, 4)
ip.close(); o_b.sync_from_device()
report("lcam_attention_gate", o_b.read_array((n,), np.int8))
f_b.close(); w_b.close(); o_b.close()

# ---- 9. head_concat_reshape ----------------------------------------------
h20, h40, h80 = (np.full(16, 1, dtype=np.int8),
                 np.full(16, 2, dtype=np.int8),
                 np.full(16, 3, dtype=np.int8))
a_b, b_b, c_b = buf(h20), buf(h40), buf(h80)
o_b = buf(fill=48)
ip = HeadConcatReshape()
ip.run(a_b.phys_addr, b_b.phys_addr, c_b.phys_addr, o_b.phys_addr)
ip.close(); o_b.sync_from_device()
report("head_concat_reshape", o_b.read_array((48,), np.int8))
a_b.close(); b_b.close(); c_b.close(); o_b.close()

print("=" * 62)
passed = sum(1 for _, ok in results if ok)
print("SUMMARY: %d/%d IPs produced output on hardware" % (passed, len(results)))
for name, ok in results:
    print("   %-24s %s" % (name, "OK" if ok else "*** FAILED ***"))
print("=" * 62)
