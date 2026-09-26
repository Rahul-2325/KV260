#!/usr/bin/env python3
"""
ip_wrappers.py
One class per HLS IP. Each opens TWO AxiLiteRegion windows:
  self.ctrl    -> "control" interface   (ap_ctrl + scalar args)
  self.ctrl_r  -> "control_r" interface (pointer args, 64-bit lo/hi)

Register offsets below are taken DIRECTLY from the Vitis-HLS-generated
xconv2d_engine_hw.h (confirmed for conv2d_engine; the same declaration-
order allocation rule applies to all other IPs since they were written
with the identical parameter style).
"""

from hw_ip_driver import AxiLiteRegion, poll_ap_done, IP_ADDR, AP_START


# ============================================================================
# conv2d_engine
# control:   H_in=0x10 W_in=0x18 C_in=0x20 C_out=0x28 KH=0x30 KW=0x38
#            stride_h=0x40 stride_w=0x48 pad_h=0x50 pad_w=0x58
#            fp_in=0x60 fp_w=0x68 fp_bias=0x70 fp_out=0x78 activation=0x80
# control_r: feat_in=0x10 weights=0x1c bias=0x28 feat_out=0x34
# ============================================================================
class Conv2dEngine:
    ACT_NONE, ACT_LEAKYRELU, ACT_HARDSIGMOID = 0, 1, 2

    def __init__(self):
        addr = IP_ADDR["conv2d_engine"]
        self.ctrl   = AxiLiteRegion(addr["control"])
        self.ctrl_r = AxiLiteRegion(addr["control_r"])

    def run(self, feat_in_addr, weights_addr, bias_addr, feat_out_addr,
            H_in, W_in, C_in, C_out, KH, KW,
            stride_h, stride_w, pad_h, pad_w,
            fp_in, fp_w, fp_bias, fp_out, activation) -> float:
        c, r = self.ctrl, self.ctrl_r
        r.write64_split(0x10, feat_in_addr)
        r.write64_split(0x1c, weights_addr)
        r.write64_split(0x28, bias_addr)
        r.write64_split(0x34, feat_out_addr)
        c.write32(0x10, H_in)
        c.write32(0x18, W_in)
        c.write32(0x20, C_in)
        c.write32(0x28, C_out)
        c.write32(0x30, KH)
        c.write32(0x38, KW)
        c.write32(0x40, stride_h)
        c.write32(0x48, stride_w)
        c.write32(0x50, pad_h)
        c.write32(0x58, pad_w)
        c.write32(0x60, fp_in)
        c.write32(0x68, fp_w)
        c.write32(0x70, fp_bias)
        c.write32(0x78, fp_out)
        c.write32(0x80, activation)
        c.write32(0x00, AP_START)
        return poll_ap_done(c, timeout_s=15.0)

    def close(self):
        self.ctrl.close()
        self.ctrl_r.close()


# ============================================================================
# depthwise_engine
# control:   H_in=0x10 W_in=0x18 C=0x20 KH=0x28 KW=0x30
#            stride_h=0x38 stride_w=0x40 pad_h=0x48 pad_w=0x50
#            fp_in=0x58 fp_w=0x60 fp_bias=0x68 fp_out=0x70
# control_r: feat_in=0x10 weights=0x1c bias=0x28 feat_out=0x34
# ============================================================================
class DepthwiseEngine:
    def __init__(self):
        addr = IP_ADDR["depthwise_engine"]
        self.ctrl   = AxiLiteRegion(addr["control"])
        self.ctrl_r = AxiLiteRegion(addr["control_r"])

    def run(self, feat_in_addr, weights_addr, bias_addr, feat_out_addr,
            H_in, W_in, C, KH, KW, stride_h, stride_w, pad_h, pad_w,
            fp_in, fp_w, fp_bias, fp_out) -> float:
        c, r = self.ctrl, self.ctrl_r
        r.write64_split(0x10, feat_in_addr)
        r.write64_split(0x1c, weights_addr)
        r.write64_split(0x28, bias_addr)
        r.write64_split(0x34, feat_out_addr)
        c.write32(0x10, H_in)
        c.write32(0x18, W_in)
        c.write32(0x20, C)
        c.write32(0x28, KH)
        c.write32(0x30, KW)
        c.write32(0x38, stride_h)
        c.write32(0x40, stride_w)
        c.write32(0x48, pad_h)
        c.write32(0x50, pad_w)
        c.write32(0x58, fp_in)
        c.write32(0x60, fp_w)
        c.write32(0x68, fp_bias)
        c.write32(0x70, fp_out)
        c.write32(0x00, AP_START)
        return poll_ap_done(c, timeout_s=15.0)

    def close(self):
        self.ctrl.close()
        self.ctrl_r.close()


# ============================================================================
# eltwise_add
# control:   H=0x10 W=0x18 C=0x20 fp_a=0x28 fp_b=0x30 fp_out=0x38
# control_r: in_a=0x10 in_b=0x1c out=0x28
# ============================================================================
class EltwiseAdd:
    def __init__(self):
        addr = IP_ADDR["eltwise_add"]
        self.ctrl   = AxiLiteRegion(addr["control"])
        self.ctrl_r = AxiLiteRegion(addr["control_r"])

    def run(self, in_a_addr, in_b_addr, out_addr, H, W, C,
            fp_a, fp_b, fp_out) -> float:
        c, r = self.ctrl, self.ctrl_r
        r.write64_split(0x10, in_a_addr)
        r.write64_split(0x1c, in_b_addr)
        r.write64_split(0x28, out_addr)
        c.write32(0x10, H)
        c.write32(0x18, W)
        c.write32(0x20, C)
        c.write32(0x28, fp_a)
        c.write32(0x30, fp_b)
        c.write32(0x38, fp_out)
        c.write32(0x00, AP_START)
        return poll_ap_done(c, timeout_s=10.0)

    def close(self):
        self.ctrl.close()
        self.ctrl_r.close()


# ============================================================================
# pool_engine
# control:   H=0x10 W=0x18 C=0x20 fp_in=0x28 fp_out=0x30
# control_r: feat_in=0x10 feat_out=0x1c
# ============================================================================
class PoolEngine:
    def __init__(self):
        addr = IP_ADDR["pool_engine"]
        self.ctrl   = AxiLiteRegion(addr["control"])
        self.ctrl_r = AxiLiteRegion(addr["control_r"])

    def run(self, feat_in_addr, feat_out_addr, H, W, C,
            fp_in, fp_out) -> float:
        c, r = self.ctrl, self.ctrl_r
        r.write64_split(0x10, feat_in_addr)
        r.write64_split(0x1c, feat_out_addr)
        c.write32(0x10, H)
        c.write32(0x18, W)
        c.write32(0x20, C)
        c.write32(0x28, fp_in)
        c.write32(0x30, fp_out)
        c.write32(0x00, AP_START)
        return poll_ap_done(c, timeout_s=10.0)

    def close(self):
        self.ctrl.close()
        self.ctrl_r.close()


# ============================================================================
# upsample_engine
# control:   H=0x10 W=0x18 C=0x20 scale=0x28
# control_r: feat_in=0x10 feat_out=0x1c
# ============================================================================
class UpsampleEngine:
    def __init__(self):
        addr = IP_ADDR["upsample_engine"]
        self.ctrl   = AxiLiteRegion(addr["control"])
        self.ctrl_r = AxiLiteRegion(addr["control_r"])

    def run(self, feat_in_addr, feat_out_addr, H, W, C, scale) -> float:
        c, r = self.ctrl, self.ctrl_r
        r.write64_split(0x10, feat_in_addr)
        r.write64_split(0x1c, feat_out_addr)
        c.write32(0x10, H)
        c.write32(0x18, W)
        c.write32(0x20, C)
        c.write32(0x28, scale)
        c.write32(0x00, AP_START)
        return poll_ap_done(c, timeout_s=10.0)

    def close(self):
        self.ctrl.close()
        self.ctrl_r.close()


# ============================================================================
# lcam_attention_gate
# control:   H=0x10 W=0x18 C=0x20 fp_feat=0x28 fp_weight=0x30 fp_out=0x38
# control_r: feat_in=0x10 weight_in=0x1c feat_out=0x28
# ============================================================================
class LcamAttentionGate:
    def __init__(self):
        addr = IP_ADDR["lcam_attention_gate"]
        self.ctrl   = AxiLiteRegion(addr["control"])
        self.ctrl_r = AxiLiteRegion(addr["control_r"])

    def run(self, feat_in_addr, weight_in_addr, feat_out_addr,
            H, W, C, fp_feat, fp_weight, fp_out) -> float:
        c, r = self.ctrl, self.ctrl_r
        r.write64_split(0x10, feat_in_addr)
        r.write64_split(0x1c, weight_in_addr)
        r.write64_split(0x28, feat_out_addr)
        c.write32(0x10, H)
        c.write32(0x18, W)
        c.write32(0x20, C)
        c.write32(0x28, fp_feat)
        c.write32(0x30, fp_weight)
        c.write32(0x38, fp_out)
        c.write32(0x00, AP_START)
        return poll_ap_done(c, timeout_s=10.0)

    def close(self):
        self.ctrl.close()
        self.ctrl_r.close()


# ============================================================================
# head_sigmoid
# control:   H=0x10 W=0x18 C=0x20 fp_in=0x28 fp_out=0x30
# control_r: data_in=0x10 data_out=0x1c
# ============================================================================
class HeadSigmoid:
    def __init__(self):
        addr = IP_ADDR["head_sigmoid"]
        self.ctrl   = AxiLiteRegion(addr["control"])
        self.ctrl_r = AxiLiteRegion(addr["control_r"])

    def run(self, data_in_addr, data_out_addr, H, W, C,
            fp_in, fp_out) -> float:
        c, r = self.ctrl, self.ctrl_r
        r.write64_split(0x10, data_in_addr)
        r.write64_split(0x1c, data_out_addr)
        c.write32(0x10, H)
        c.write32(0x18, W)
        c.write32(0x20, C)
        c.write32(0x28, fp_in)
        c.write32(0x30, fp_out)
        c.write32(0x00, AP_START)
        return poll_ap_done(c, timeout_s=10.0)

    def close(self):
        self.ctrl.close()
        self.ctrl_r.close()


# ============================================================================
# head_transpose
# control:   H=0x10 W=0x18 C=0x20
# control_r: data_in=0x10 data_out=0x1c
#
# NOTE: This reflects the ORIGINAL two-interface (control/control_r) register
# layout. head_transpose.cpp in hls_ip_sources/ was later modified to a
# single-interface layout as a debugging experiment (see PROJECT_HISTORY.md
# §23) -- that fix was tested and RULED OUT as the root cause, so it does not
# need to be propagated here unless you're specifically re-testing that exact
# experiment, in which case use debug_scripts/test_head_transpose_v2.py's
# register scheme instead (single CTRL_BASE, no control_r).
# ============================================================================
class HeadTranspose:
    def __init__(self):
        addr = IP_ADDR["head_transpose"]
        self.ctrl   = AxiLiteRegion(addr["control"])
        self.ctrl_r = AxiLiteRegion(addr["control_r"])

    def run(self, data_in_addr, data_out_addr, H, W, C) -> float:
        c, r = self.ctrl, self.ctrl_r
        r.write64_split(0x10, data_in_addr)
        r.write64_split(0x1c, data_out_addr)
        c.write32(0x10, H)
        c.write32(0x18, W)
        c.write32(0x20, C)
        c.write32(0x00, AP_START)
        return poll_ap_done(c, timeout_s=10.0)

    def close(self):
        self.ctrl.close()
        self.ctrl_r.close()


# ============================================================================
# head_concat_reshape
# control:   (no scalar args)
# control_r: head_20=0x10 head_40=0x1c head_80=0x28 output=0x34
# ============================================================================
class HeadConcatReshape:
    def __init__(self):
        addr = IP_ADDR["head_concat_reshape"]
        self.ctrl   = AxiLiteRegion(addr["control"])
        self.ctrl_r = AxiLiteRegion(addr["control_r"])

    def run(self, head_20_addr, head_40_addr, head_80_addr,
            output_addr) -> float:
        c, r = self.ctrl, self.ctrl_r
        r.write64_split(0x10, head_20_addr)
        r.write64_split(0x1c, head_40_addr)
        r.write64_split(0x28, head_80_addr)
        r.write64_split(0x34, output_addr)
        c.write32(0x00, AP_START)
        return poll_ap_done(c, timeout_s=10.0)

    def close(self):
        self.ctrl.close()
        self.ctrl_r.close()
