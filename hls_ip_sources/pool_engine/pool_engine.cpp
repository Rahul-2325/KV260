#include "pool_engine.h"

// ============================================================
// pool_engine — global adaptive average pooling
// Covers: pool-fix ops in DPU subgraphs [01],[03],[05],[07]
// Used in LCAM channel attention: [H,W,C] → [1,1,C]
// ============================================================

void pool_engine(
    data_t* feat_in,
    data_t* feat_out,
    int H, int W, int C,
    int fp_in, int fp_out)
{
    #pragma HLS INTERFACE m_axi port=feat_in  bundle=gmem0 offset=slave depth=MAX_POOL_FEAT
    #pragma HLS INTERFACE m_axi port=feat_out bundle=gmem1 offset=slave depth=512
    #pragma HLS INTERFACE s_axilite port=H      bundle=control
    #pragma HLS INTERFACE s_axilite port=W      bundle=control
    #pragma HLS INTERFACE s_axilite port=C      bundle=control
    #pragma HLS INTERFACE s_axilite port=fp_in  bundle=control
    #pragma HLS INTERFACE s_axilite port=fp_out bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    int spatial = H * W;

    for (int c = 0; c < C; c++) {
        #pragma HLS PIPELINE II=1

        int sum = 0;
        for (int i = 0; i < spatial; i++) {
            sum += (int)feat_in[i * C + c];
        }

        // Average: divide by spatial size
        // result in same fix_point as input
        int avg = sum / spatial;

        // Shift to output fix_point
        int shift = fp_out - fp_in;
        if      (shift > 0) avg = avg << shift;
        else if (shift < 0) avg = avg >> (-shift);

        // Saturate
        if      (avg >  127) avg =  127;
        else if (avg < -128) avg = -128;

        feat_out[c] = (data_t)avg;
    }
}
