#include "head_sigmoid.h"
#include <cmath>

// ============================================================
// head_sigmoid
//
// PURPOSE:
//   Applies sigmoid activation to INT8 tensors.
//   Covers CPU subgraphs [10],[11],[12] (objectness, C=1)
//   and [13],[16],[19] (class scores, C=2) in lcam_v5.xmodel.
//
// OPERATION per element:
//   1. Dequantize: x_f = x_int8 * 2^(-fp_in)
//   2. Sigmoid:    s   = 1 / (1 + exp(-x_f))
//   3. Requantize: out = clip(round(s * 2^fp_out), -128, 127)
//
// INSTANCES from xmodel:
//   [10]: H=20, W=20, C=1,  fp_in=5, fp_out=5  objectness 20x20
//   [11]: H=40, W=40, C=1,  fp_in=5, fp_out=5  objectness 40x40
//   [12]: H=80, W=80, C=1,  fp_in=5, fp_out=5  objectness 80x80
//   [13]: H=20, W=20, C=2,  fp_in=5, fp_out=5  class 20x20
//   [16]: H=40, W=40, C=2,  fp_in=5, fp_out=5  class 40x40
//   [19]: H=80, W=80, C=2,  fp_in=5, fp_out=5  class 80x80
// ============================================================

void head_sigmoid(
    data_t*  data_in,
    data_t*  data_out,
    int      H,
    int      W,
    int      C,
    int      fp_in,
    int      fp_out
)
{
    #pragma HLS INTERFACE m_axi port=data_in  bundle=gmem0 \
                                offset=slave  depth=12800
    #pragma HLS INTERFACE m_axi port=data_out bundle=gmem1 \
                                offset=slave  depth=12800

    #pragma HLS INTERFACE s_axilite port=H       bundle=control
    #pragma HLS INTERFACE s_axilite port=W       bundle=control
    #pragma HLS INTERFACE s_axilite port=C       bundle=control
    #pragma HLS INTERFACE s_axilite port=fp_in   bundle=control
    #pragma HLS INTERFACE s_axilite port=fp_out  bundle=control
    #pragma HLS INTERFACE s_axilite port=return  bundle=control

    float scale_in  = 1.0f / (float)(1 << fp_in);
    float scale_out = (float)(1 << fp_out);

    int total = H * W * C;

    for (int i = 0; i < total; i++) {
        #pragma HLS PIPELINE II=1

        // Dequantize
        float x = (float)(int)data_in[i] * scale_in;

        // Sigmoid: 1 / (1 + e^-x)
        // hls::exp() maps to hardware-optimized exp unit in DSPs
        float s = 1.0f / (1.0f + hls::exp(-x));

        // Requantize
        int q = (int)roundf(s * scale_out);
        if      (q >  127) q =  127;
        else if (q < -128) q = -128;

        data_out[i] = (data_t)q;
    }
}
