#include "conv2d_engine.h"
#include <cmath>

// ============================================================
// conv2d_engine.cpp
//
// Parameterized INT8 convolution engine for LCAM-YOLOX.
// Covers all 111 conv2d-fix layers from lcam_v5.xmodel.
//
// QUANTIZATION MATH:
//   real_value = int8_value * 2^(-fix_point)
//
//   For conv: out_real = sum(in_real * w_real) + bias_real
//   In INT8:  acc = sum(in_int8 * w_int8)
//   Scale:    out_int8 = round(acc * 2^(fp_in + fp_w - fp_out)
//                             + bias_int8 * 2^(fp_bias - fp_out))
//
// WEIGHT LAYOUT (matches xmodel const-fix tensors):
//   weights[co][kh][kw][ci] — OHWI layout
//   Index: co*KH*KW*C_in + kh*KW*C_in + kw*C_in + ci
//
// INPUT/OUTPUT LAYOUT: NHWC
//   feat_in[h][w][ci]  = feat_in[(h*W_in + w)*C_in + ci]
//   feat_out[h][w][co] = feat_out[(h*W_out + w)*C_out + co]
// ============================================================

// Apply activation function to INT32 accumulator, return INT8
static data_t apply_activation(acc_t acc_scaled, int activation)
{
    #pragma HLS INLINE
    int val = (int)acc_scaled;

    if (activation == ACT_LEAKYRELU) {
        // LeakyReLU: negative_slope = 0.1 = 1/10
        if (val < 0) val = val / 10;
    } else if (activation == ACT_HARDSIGMOID) {
        // HardSigmoid: clip((x/6) + 0.5, 0, 1) — applied in float
        float fval = (float)val / 128.0f;   // rough dequant
        fval = fval / 6.0f + 0.5f;
        if      (fval < 0.0f) fval = 0.0f;
        else if (fval > 1.0f) fval = 1.0f;
        val = (int)(fval * 128.0f);
    }

    // Saturate to INT8
    if      (val >  127) val =  127;
    else if (val < -128) val = -128;
    return (data_t)val;
}

void conv2d_engine(
    data_t*  feat_in,
    data_t*  weights,
    data_t*  bias,
    data_t*  feat_out,
    int      H_in,
    int      W_in,
    int      C_in,
    int      C_out,
    int      KH,
    int      KW,
    int      stride_h,
    int      stride_w,
    int      pad_h,
    int      pad_w,
    int      fp_in,
    int      fp_w,
    int      fp_bias,
    int      fp_out,
    int      activation
)
{
    // ── AXI master interfaces ─────────────────────────────────
    #pragma HLS INTERFACE m_axi port=feat_in  bundle=gmem0 offset=slave depth=MAX_FEAT
    #pragma HLS INTERFACE m_axi port=weights  bundle=gmem1 offset=slave depth=MAX_WEIGHT
    #pragma HLS INTERFACE m_axi port=bias     bundle=gmem2 offset=slave depth=MAX_BIAS
    #pragma HLS INTERFACE m_axi port=feat_out bundle=gmem3 offset=slave depth=MAX_OUT

    // ── AXI-Lite control ──────────────────────────────────────
    #pragma HLS INTERFACE s_axilite port=H_in       bundle=control
    #pragma HLS INTERFACE s_axilite port=W_in       bundle=control
    #pragma HLS INTERFACE s_axilite port=C_in       bundle=control
    #pragma HLS INTERFACE s_axilite port=C_out      bundle=control
    #pragma HLS INTERFACE s_axilite port=KH         bundle=control
    #pragma HLS INTERFACE s_axilite port=KW         bundle=control
    #pragma HLS INTERFACE s_axilite port=stride_h   bundle=control
    #pragma HLS INTERFACE s_axilite port=stride_w   bundle=control
    #pragma HLS INTERFACE s_axilite port=pad_h      bundle=control
    #pragma HLS INTERFACE s_axilite port=pad_w      bundle=control
    #pragma HLS INTERFACE s_axilite port=fp_in      bundle=control
    #pragma HLS INTERFACE s_axilite port=fp_w       bundle=control
    #pragma HLS INTERFACE s_axilite port=fp_bias    bundle=control
    #pragma HLS INTERFACE s_axilite port=fp_out     bundle=control
    #pragma HLS INTERFACE s_axilite port=activation bundle=control
    #pragma HLS INTERFACE s_axilite port=return     bundle=control

    // ── Output spatial dimensions ─────────────────────────────
    int H_out = (H_in + 2*pad_h - KH) / stride_h + 1;
    int W_out = (W_in + 2*pad_w - KW) / stride_w + 1;

    // ── Quantization scale factor ─────────────────────────────
    // acc is in units of 2^(-(fp_in + fp_w))
    // output is in units of 2^(-fp_out)
    // so shift = fp_in + fp_w - fp_out
    int shift = fp_in + fp_w - fp_out;

    // ── Main convolution loops ────────────────────────────────
    for (int h = 0; h < H_out; h++) {
        for (int w = 0; w < W_out; w++) {
            for (int co = 0; co < C_out; co++) {
                #pragma HLS PIPELINE II=1

                acc_t acc = 0;

                // Kernel loops
                for (int kh = 0; kh < KH; kh++) {
                    for (int kw = 0; kw < KW; kw++) {
                        // Input spatial position
                        int ih = h * stride_h - pad_h + kh;
                        int iw = w * stride_w - pad_w + kw;

                        // Zero padding check
                        if (ih < 0 || ih >= H_in || iw < 0 || iw >= W_in)
                            continue;

                        // Accumulate over input channels
                        for (int ci = 0; ci < C_in; ci++) {
                            int feat_idx   = (ih * W_in + iw) * C_in + ci;
                            int weight_idx = ((co * KH + kh) * KW + kw) * C_in + ci;

                            acc += (acc_t)feat_in[feat_idx]
                                 * (acc_t)weights[weight_idx];
                        }
                    }
                }

                // Add bias (bias is in units of 2^(-fp_bias))
                // Shift bias to match accumulator scale
                int bias_shift = fp_bias - fp_out;
                acc_t bias_scaled;
                if (bias_shift >= 0)
                    bias_scaled = (acc_t)bias[co] * (acc_t)(1 << bias_shift);
                else
                    bias_scaled = (acc_t)bias[co] >> (-bias_shift);

                // Shift accumulator to output scale
                acc_t acc_out;
                if (shift >= 0)
                    acc_out = (acc >> shift) + bias_scaled;
                else
                    acc_out = (acc << (-shift)) + bias_scaled;

                // Apply activation and write output
                int out_idx = (h * W_out + w) * C_out + co;
                feat_out[out_idx] = apply_activation(acc_out, activation);
            }
        }
    }
}
