#ifndef DEPTHWISE_ENGINE_H
#define DEPTHWISE_ENGINE_H
#include "ap_int.h"
typedef ap_int<8>  data_t;
typedef ap_int<32> acc_t;
#define MAX_DW_FEAT   (80 * 80 * 128)
#define MAX_DW_WEIGHT (128 * 3 * 3)

// Depthwise conv: each input channel has its own KxK filter
// C_in == C_out == C (groups == C)
void depthwise_engine(
    data_t* feat_in,   // [H * W * C]
    data_t* weights,   // [C * KH * KW]
    data_t* bias,      // [C]
    data_t* feat_out,  // [H_out * W_out * C]
    int     H_in,
    int     W_in,
    int     C,
    int     KH,
    int     KW,
    int     stride_h,
    int     stride_w,
    int     pad_h,
    int     pad_w,
    int     fp_in,
    int     fp_w,
    int     fp_bias,
    int     fp_out
);
#endif
