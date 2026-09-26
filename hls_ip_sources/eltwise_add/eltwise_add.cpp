#include "eltwise_add.h"
#include <cmath>

// ============================================================
// eltwise_add — residual connection (add two feature maps)
// Covers: eltwise-fix ops in DPU subgraphs
// Both inputs may have different fix_points — dequant, add, requant
// ============================================================

void eltwise_add(
    data_t* in_a,
    data_t* in_b,
    data_t* out,
    int H, int W, int C,
    int fp_a, int fp_b, int fp_out)
{
    #pragma HLS INTERFACE m_axi port=in_a bundle=gmem0 offset=slave depth=MAX_ELTWISE
    #pragma HLS INTERFACE m_axi port=in_b bundle=gmem1 offset=slave depth=MAX_ELTWISE
    #pragma HLS INTERFACE m_axi port=out  bundle=gmem2 offset=slave depth=MAX_ELTWISE
    #pragma HLS INTERFACE s_axilite port=H      bundle=control
    #pragma HLS INTERFACE s_axilite port=W      bundle=control
    #pragma HLS INTERFACE s_axilite port=C      bundle=control
    #pragma HLS INTERFACE s_axilite port=fp_a   bundle=control
    #pragma HLS INTERFACE s_axilite port=fp_b   bundle=control
    #pragma HLS INTERFACE s_axilite port=fp_out bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    int total = H * W * C;

    // Align both inputs to output fix_point before adding
    // real_a = a * 2^(-fp_a), real_b = b * 2^(-fp_b)
    // out_int = round((real_a + real_b) * 2^fp_out)
    //         = round(a * 2^(fp_out-fp_a) + b * 2^(fp_out-fp_b))

    int shift_a = fp_out - fp_a;
    int shift_b = fp_out - fp_b;

    for (int i = 0; i < total; i++) {
        #pragma HLS PIPELINE II=1

        int va = (int)in_a[i];
        int vb = (int)in_b[i];

        // Shift to output scale
        int sa = (shift_a >= 0) ? (va << shift_a) : (va >> (-shift_a));
        int sb = (shift_b >= 0) ? (vb << shift_b) : (vb >> (-shift_b));

        int result = sa + sb;

        // Saturate to INT8
        if      (result >  127) result =  127;
        else if (result < -128) result = -128;

        out[i] = (data_t)result;
    }
}
