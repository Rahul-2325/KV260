#ifndef CONV2D_ENGINE_H
#define CONV2D_ENGINE_H

#include "ap_int.h"
#include "hls_math.h"

typedef ap_int<8>   data_t;
typedef ap_int<32>  acc_t;

#define MAX_H       320
#define MAX_W       320
#define MAX_C_IN    512
#define MAX_C_OUT   512
#define MAX_K       7
#define MAX_FEAT    (MAX_H * MAX_W * MAX_C_IN)
#define MAX_WEIGHT  (MAX_C_OUT * MAX_K * MAX_K * MAX_C_IN)
#define MAX_BIAS    MAX_C_OUT
#define MAX_OUT     (MAX_H * MAX_W * MAX_C_OUT)

#define ACT_NONE        0
#define ACT_LEAKYRELU   1
#define ACT_HARDSIGMOID 2

void conv2d_engine(
    data_t*  feat_in,
    data_t*  weights,
    data_t*  bias,
    data_t*  feat_out,
    int      H_in,
    int      W_in,
    int      C_in,
    int      C_out,
    int      KH,
    int      KW,
    int      stride_h,
    int      stride_w,
    int      pad_h,
    int      pad_w,
    int      fp_in,
    int      fp_w,
    int      fp_bias,
    int      fp_out,
    int      activation
);

#endif
