#ifndef POOL_ENGINE_H
#define POOL_ENGINE_H
#include "ap_int.h"
typedef ap_int<8> data_t;
#define MAX_POOL_FEAT (160 * 160 * 512)

// Adaptive Average Pooling → output [1, 1, 1, C]
// Used in LCAM channel attention (pool-fix ops)
void pool_engine(
    data_t* feat_in,   // [H * W * C]
    data_t* feat_out,  // [1 * 1 * C] global average
    int     H,
    int     W,
    int     C,
    int     fp_in,
    int     fp_out
);
#endif
