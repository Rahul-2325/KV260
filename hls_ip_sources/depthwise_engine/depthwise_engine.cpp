#include "depthwise_engine.h"

// ============================================================
// depthwise_engine — depthwise convolution
// One filter per channel, no cross-channel mixing
// Covers: depthwise-fix ops in DPU subgraphs
// ============================================================

void depthwise_engine(
    data_t* feat_in,
    data_t* weights,
    data_t* bias,
    data_t* feat_out,
    int H_in, int W_in, int C,
    int KH, int KW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int fp_in, int fp_w, int fp_bias, int fp_out)
{
    #pragma HLS INTERFACE m_axi port=feat_in  bundle=gmem0 offset=slave depth=MAX_DW_FEAT
    #pragma HLS INTERFACE m_axi port=weights  bundle=gmem1 offset=slave depth=MAX_DW_WEIGHT
    #pragma HLS INTERFACE m_axi port=bias     bundle=gmem2 offset=slave depth=128
    #pragma HLS INTERFACE m_axi port=feat_out bundle=gmem3 offset=slave depth=MAX_DW_FEAT
    #pragma HLS INTERFACE s_axilite port=H_in     bundle=control
    #pragma HLS INTERFACE s_axilite port=W_in     bundle=control
    #pragma HLS INTERFACE s_axilite port=C        bundle=control
    #pragma HLS INTERFACE s_axilite port=KH       bundle=control
    #pragma HLS INTERFACE s_axilite port=KW       bundle=control
    #pragma HLS INTERFACE s_axilite port=stride_h bundle=control
    #pragma HLS INTERFACE s_axilite port=stride_w bundle=control
    #pragma HLS INTERFACE s_axilite port=pad_h    bundle=control
    #pragma HLS INTERFACE s_axilite port=pad_w    bundle=control
    #pragma HLS INTERFACE s_axilite port=fp_in    bundle=control
    #pragma HLS INTERFACE s_axilite port=fp_w     bundle=control
    #pragma HLS INTERFACE s_axilite port=fp_bias  bundle=control
    #pragma HLS INTERFACE s_axilite port=fp_out   bundle=control
    #pragma HLS INTERFACE s_axilite port=return   bundle=control

    int H_out = (H_in + 2*pad_h - KH) / stride_h + 1;
    int W_out = (W_in + 2*pad_w - KW) / stride_w + 1;
    int shift = fp_in + fp_w - fp_out;

    for (int h = 0; h < H_out; h++) {
        for (int w = 0; w < W_out; w++) {
            for (int c = 0; c < C; c++) {
                #pragma HLS PIPELINE II=1

                acc_t acc = 0;
                for (int kh = 0; kh < KH; kh++) {
                    for (int kw = 0; kw < KW; kw++) {
                        int ih = h*stride_h - pad_h + kh;
                        int iw = w*stride_w - pad_w + kw;
                        if (ih < 0 || ih >= H_in || iw < 0 || iw >= W_in)
                            continue;
                        int fi = (ih*W_in + iw)*C + c;
                        int wi = (c*KH + kh)*KW + kw;
                        acc += (acc_t)feat_in[fi] * (acc_t)weights[wi];
                    }
                }

                // Add bias
                int bias_shift = fp_bias - fp_out;
                acc_t b = (bias_shift >= 0)
                    ? (acc_t)bias[c] * (acc_t)(1 << bias_shift)
                    : (acc_t)bias[c] >> (-bias_shift);

                acc_t out_val;
                if (shift >= 0) out_val = (acc >> shift) + b;
                else            out_val = (acc << (-shift)) + b;

                // LeakyReLU (depthwise in YOLOX always has activation)
                int v = (int)out_val;
                if (v < 0) v = v / 10;
                if      (v >  127) v =  127;
                else if (v < -128) v = -128;

                feat_out[(h*W_out + w)*C + c] = (data_t)v;
            }
        }
    }
}
