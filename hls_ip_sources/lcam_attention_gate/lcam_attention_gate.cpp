#include "lcam_attention_gate.h"
#include <cmath>    // for roundf

// ============================================================
// lcam_attention_gate
//
// PURPOSE:
//   Implements the LCAM attention gate operation that sits
//   between DPU subgraphs in LCAM-YOLOX. Corresponds to
//   CPU subgraphs [02], [04], [06], [08] in lcam_v5.xmodel.
//
// OPERATION:
//   For every spatial position (h,w) and channel c:
//     1. Dequantize:  feat_f   = feat_in[h,w,c]   * 2^(-fp_feat)
//     2. Dequantize:  weight_f = weight_in[h,w,0]  * 2^(-fp_weight)
//     3. Gate apply:  result_f = feat_f * weight_f
//     4. Requantize:  feat_out[h,w,c] = clip(round(result_f * 2^fp_out), -128, 127)
//
// TENSOR LAYOUT:  NHWC  (N=1 always)
//
// INSTANCES in xmodel:
//   Subgraph [02]: H=160, W=160, C=64,  fp_feat=4, fp_weight=7, fp_out=5
//   Subgraph [04]: H=80,  W=80,  C=128, fp_feat=5, fp_weight=7, fp_out=5
//   Subgraph [06]: H=40,  W=40,  C=256, fp_feat=5, fp_weight=7, fp_out=5
//   Subgraph [08]: H=20,  W=20,  C=512, fp_feat=5, fp_weight=7, fp_out=6
// ============================================================

void lcam_attention_gate(
    data_t*  feat_in,
    data_t*  weight_in,
    data_t*  feat_out,
    int      H,
    int      W,
    int      C,
    int      fp_feat,
    int      fp_weight,
    int      fp_out
)
{
    // --------------------------------------------------------
    // AXI master interfaces for DDR access
    // Three separate bundles so all three arrays can be
    // accessed in parallel without port arbitration conflicts
    // --------------------------------------------------------
    #pragma HLS INTERFACE m_axi port=feat_in   bundle=gmem0 \
                                offset=slave   depth=13107200
    #pragma HLS INTERFACE m_axi port=weight_in bundle=gmem1 \
                                offset=slave   depth=25600
    #pragma HLS INTERFACE m_axi port=feat_out  bundle=gmem2 \
                                offset=slave   depth=13107200

    // AXI-Lite slave for all scalar control registers + ap_ctrl
    #pragma HLS INTERFACE s_axilite port=H         bundle=control
    #pragma HLS INTERFACE s_axilite port=W         bundle=control
    #pragma HLS INTERFACE s_axilite port=C         bundle=control
    #pragma HLS INTERFACE s_axilite port=fp_feat   bundle=control
    #pragma HLS INTERFACE s_axilite port=fp_weight bundle=control
    #pragma HLS INTERFACE s_axilite port=fp_out    bundle=control
    #pragma HLS INTERFACE s_axilite port=return    bundle=control

    // --------------------------------------------------------
    // Pre-compute scale factors from fix_point values
    // fix_point=N means: real_value = int8_value * 2^(-N)
    // So scale_in  = 2^(-fp)  = 1.0 / (1 << fp)
    //    scale_out = 2^(fp)   = (float)(1 << fp)
    // --------------------------------------------------------
    float scale_feat   = 1.0f / (float)(1 << fp_feat);
    float scale_weight = 1.0f / (float)(1 << fp_weight);
    float scale_out    = (float)(1 << fp_out);

    // --------------------------------------------------------
    // Main compute loop — iterate over every spatial pixel
    // and every channel
    // --------------------------------------------------------
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {

            // Load the attention weight for this (h,w) position
            // Weight tensor layout: [H, W, 1] — one scalar per pixel
            int weight_idx      = h * W + w;
            float weight_float  = (float)(int)weight_in[weight_idx]
                                  * scale_weight;

            // Apply attention weight to all C channels at this position
            for (int c = 0; c < C; c++) {
                #pragma HLS PIPELINE II=1

                // NHWC index: row-major, channel is innermost
                int feat_idx = (h * W + w) * C + c;

                // Step 1: Dequantize feature
                float feat_float = (float)(int)feat_in[feat_idx]
                                   * scale_feat;

                // Step 2: Element-wise multiply (attention gate)
                float result = feat_float * weight_float;

                // Step 3: Requantize to INT8 at output fix_point
                float scaled = result * scale_out;
                int   q      = (int)roundf(scaled);

                // Step 4: Saturate to INT8 range [-128, 127]
                if      (q >  127) q =  127;
                else if (q < -128) q = -128;

                feat_out[feat_idx] = (data_t)q;
            }
        }
    }
}
