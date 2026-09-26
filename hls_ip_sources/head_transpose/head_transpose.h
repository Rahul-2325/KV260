#ifndef HEAD_TRANSPOSE_H
#define HEAD_TRANSPOSE_H

#include "ap_int.h"

typedef ap_int<8>  data_t;

// Largest case: [1, 7, 80, 80] = 44800 elements
#define MAX_TRANSPOSE_ELEMENTS  44800

// ============================================================
// Converts NCHW layout → NHWC layout
// Input:  [1, C, H, W]  →  data_in[c * H*W + h*W + w]
// Output: [1, H, W, C]  →  data_out[h * W*C + w*C + c]
//
// Instances from xmodel:
//   [15]: [1,7,20,20] → [1,20,20,7]   output_fix sink transpose 20x20
//   [18]: [1,7,40,40] → [1,40,40,7]   output_fix sink transpose 40x40
//   [21]: [1,7,80,80] → [1,80,80,7]   output_fix sink transpose 80x80
// ============================================================
void head_transpose(
    data_t*  data_in,    // [C * H * W]  NCHW layout
    data_t*  data_out,   // [H * W * C]  NHWC layout
    int      H,
    int      W,
    int      C           // always 7 (4 bbox + 1 obj + 2 cls)
);

#endif // HEAD_TRANSPOSE_H
