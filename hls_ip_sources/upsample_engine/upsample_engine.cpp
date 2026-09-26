#include "upsample_engine.h"

// ============================================================
// upsample_engine — nearest neighbor 2x upsample
// Used in FPN/PAN neck of YOLOX to merge feature scales
// ============================================================

void upsample_engine(
    data_t* feat_in,
    data_t* feat_out,
    int H, int W, int C, int scale)
{
    #pragma HLS INTERFACE m_axi port=feat_in  bundle=gmem0 offset=slave depth=MAX_UPSAMPLE
    #pragma HLS INTERFACE m_axi port=feat_out bundle=gmem1 offset=slave depth=MAX_UPSAMPLE*4
    #pragma HLS INTERFACE s_axilite port=H      bundle=control
    #pragma HLS INTERFACE s_axilite port=W      bundle=control
    #pragma HLS INTERFACE s_axilite port=C      bundle=control
    #pragma HLS INTERFACE s_axilite port=scale  bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    int H_out = H * scale;
    int W_out = W * scale;

    for (int h = 0; h < H_out; h++) {
        for (int w = 0; w < W_out; w++) {
            // Nearest neighbor: map output pixel to input pixel
            int src_h = h / scale;
            int src_w = w / scale;

            for (int c = 0; c < C; c++) {
                #pragma HLS PIPELINE II=1
                int src_idx = (src_h * W + src_w) * C + c;
                int dst_idx = (h * W_out + w) * C + c;
                feat_out[dst_idx] = feat_in[src_idx];
            }
        }
    }
}
