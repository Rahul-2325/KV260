#ifndef ELTWISE_ADD_H
#define ELTWISE_ADD_H
#include "ap_int.h"
typedef ap_int<8> data_t;
#define MAX_ELTWISE (160 * 160 * 64)

void eltwise_add(
    data_t* in_a,    // [H * W * C] first input
    data_t* in_b,    // [H * W * C] second input
    data_t* out,     // [H * W * C] output
    int     H,
    int     W,
    int     C,
    int     fp_a,    // fix_point of in_a
    int     fp_b,    // fix_point of in_b
    int     fp_out   // fix_point of output
);
#endif
