#ifndef HEAD_SIGMOID_H
#define HEAD_SIGMOID_H

#include "ap_int.h"
#include "hls_math.h"   // for hls::exp() — hardware-optimized exp

typedef ap_int<8>  data_t;

// ============================================================
// Largest case: subgraph [12] → [1, 80, 80, 2] = 12800 elements
// ============================================================
#define MAX_SIGMOID_ELEMENTS  12800

void head_sigmoid(
    data_t*  data_in,    // [H * W * C]  INT8 from DPU
    data_t*  data_out,   // [H * W * C]  INT8 back to DPU
    int      H,
    int      W,
    int      C,          // 1 (objectness) or 2 (class scores)
    int      fp_in,      // fix_point of input  (always 5 from xmodel)
    int      fp_out      // fix_point of output (always 5 from xmodel)
);

#endif // HEAD_SIGMOID_H
