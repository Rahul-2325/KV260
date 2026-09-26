#ifndef HEAD_CONCAT_RESHAPE_H
#define HEAD_CONCAT_RESHAPE_H

#include "ap_int.h"

typedef ap_int<8>  data_t;

// Total anchors: 20*20 + 40*40 + 80*80 = 400 + 1600 + 6400 = 8400
#define ANCHORS_20   400
#define ANCHORS_40   1600
#define ANCHORS_80   6400
#define ANCHORS_TOTAL  (ANCHORS_20 + ANCHORS_40 + ANCHORS_80)  // 8400
#define NUM_CLASSES  7    // 4 bbox + 1 obj + 2 cls

// ============================================================
// Concatenates 3 detection head outputs along the anchor dim
// then reshapes into [1, 8400, 7] layout for decode_0 IP.
//
// Input tensors (all NHWC, all INT8 fp=5):
//   head_20: [1, 20, 20, 7]   = 2800 bytes
//   head_40: [1, 40, 40, 7]   = 11200 bytes
//   head_80: [1, 80, 80, 7]   = 44800 bytes
//
// Output: [1, 8400, 7] = 58800 bytes (contiguous, anchor-major)
//   First 400×7  = head_20 flattened
//   Next  1600×7 = head_40 flattened
//   Last  6400×7 = head_80 flattened
// ============================================================
void head_concat_reshape(
    data_t*  head_20,    // [20*20*7]  = 2800  elements
    data_t*  head_40,    // [40*40*7]  = 11200 elements
    data_t*  head_80,    // [80*80*7]  = 44800 elements
    data_t*  output      // [8400*7]   = 58800 elements
);

#endif // HEAD_CONCAT_RESHAPE_H
