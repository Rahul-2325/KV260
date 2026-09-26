#include "conv2d_engine.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

// ============================================================
// conv2d_engine testbench
// Tests 6 real configurations from lcam_v5.xmodel
// ============================================================

// Software golden reference
static void golden_conv2d(
    int8_t* feat_in, int8_t* weights, int8_t* bias, int8_t* feat_out,
    int H_in, int W_in, int C_in, int C_out,
    int KH, int KW, int stride_h, int stride_w,
    int pad_h, int pad_w,
    int fp_in, int fp_w, int fp_bias, int fp_out,
    int activation)
{
    int H_out = (H_in + 2*pad_h - KH) / stride_h + 1;
    int W_out = (W_in + 2*pad_w - KW) / stride_w + 1;
    int shift = fp_in + fp_w - fp_out;

    for (int h = 0; h < H_out; h++) {
        for (int w = 0; w < W_out; w++) {
            for (int co = 0; co < C_out; co++) {
                int32_t acc = 0;
                for (int kh = 0; kh < KH; kh++) {
                    for (int kw = 0; kw < KW; kw++) {
                        int ih = h*stride_h - pad_h + kh;
                        int iw = w*stride_w - pad_w + kw;
                        if (ih < 0 || ih >= H_in || iw < 0 || iw >= W_in)
                            continue;
                        for (int ci = 0; ci < C_in; ci++) {
                            int fi = (ih*W_in + iw)*C_in + ci;
                            int wi = ((co*KH + kh)*KW + kw)*C_in + ci;
                            acc += (int32_t)feat_in[fi] * (int32_t)weights[wi];
                        }
                    }
                }
                // Bias
                int bias_shift = fp_bias - fp_out;
                int32_t b = (bias_shift >= 0)
                    ? (int32_t)bias[co] * (1 << bias_shift)
                    : (int32_t)bias[co] >> (-bias_shift);
                int32_t out_val;
                if (shift >= 0) out_val = (acc >> shift) + b;
                else            out_val = (acc << (-shift)) + b;

                // Activation
                if (activation == 1 && out_val < 0) out_val = out_val / 10;

                // Saturate
                if      (out_val >  127) out_val =  127;
                else if (out_val < -128) out_val = -128;

                feat_out[(h*W_out + w)*C_out + co] = (int8_t)out_val;
            }
        }
    }
}

static int run_test(
    int id, const char* desc,
    int H_in, int W_in, int C_in, int C_out,
    int KH, int KW, int stride_h, int stride_w,
    int pad_h, int pad_w,
    int fp_in, int fp_w, int fp_bias, int fp_out,
    int activation)
{
    int H_out = (H_in + 2*pad_h - KH) / stride_h + 1;
    int W_out = (W_in + 2*pad_w - KW) / stride_w + 1;

    int n_in  = H_in  * W_in  * C_in;
    int n_w   = C_out * KH * KW * C_in;
    int n_b   = C_out;
    int n_out = H_out * W_out * C_out;

    int8_t* feat_in  = new int8_t[n_in];
    int8_t* wts      = new int8_t[n_w];
    int8_t* b        = new int8_t[n_b];
    int8_t* out_hls  = new int8_t[n_out];
    int8_t* out_ref  = new int8_t[n_out];

    // Fill with deterministic patterns
    for (int i = 0; i < n_in; i++) feat_in[i] = (int8_t)((i % 100) - 50);
    for (int i = 0; i < n_w;  i++) wts[i]     = (int8_t)((i % 60)  - 30);
    for (int i = 0; i < n_b;  i++) b[i]        = (int8_t)((i % 20)  - 10);

    // Golden reference
    golden_conv2d(feat_in, wts, b, out_ref,
                  H_in, W_in, C_in, C_out,
                  KH, KW, stride_h, stride_w, pad_h, pad_w,
                  fp_in, fp_w, fp_bias, fp_out, activation);

    // HLS DUT
    conv2d_engine(
        (data_t*)feat_in, (data_t*)wts, (data_t*)b, (data_t*)out_hls,
        H_in, W_in, C_in, C_out,
        KH, KW, stride_h, stride_w, pad_h, pad_w,
        fp_in, fp_w, fp_bias, fp_out, activation);

    // Compare — allow ±1 for rounding
    int mismatches = 0;
    for (int i = 0; i < n_out; i++) {
        if (abs((int)out_hls[i] - (int)out_ref[i]) > 1) {
            mismatches++;
            if (mismatches <= 3)
                printf("  MISMATCH[%d]: HLS=%d REF=%d\n",
                       i, (int)out_hls[i], (int)out_ref[i]);
        }
    }

    printf("[Test %d] %-35s H=%d W=%d Ci=%d Co=%d K=%dx%d S=%d  -->  ",
           id, desc, H_in, W_in, C_in, C_out, KH, KW, stride_h);
    if (mismatches == 0)
        printf("PASS (%d elements)\n", n_out);
    else
        printf("FAIL (%d mismatches)\n", mismatches);

    delete[] feat_in; delete[] wts; delete[] b;
    delete[] out_hls; delete[] out_ref;
    return mismatches;
}

int main()
{
    printf("============================================================\n");
    printf("conv2d_engine Testbench — Real layer configs from lcam_v5\n");
    printf("============================================================\n\n");

    int fails = 0;

    // Conv#1:  stem 3x3 stride2  [1,640,640,3]→[1,320,320,32]
    fails += run_test(1, "stem 3x3 stride2",
        640, 640, 3, 32, 3, 3, 2, 2, 1, 1,
        3, 4, 7, 3, ACT_LEAKYRELU);

    // Conv#2:  Dark1 3x3 stride2 [1,320,320,32]→[1,160,160,64]
    fails += run_test(2, "dark1 3x3 stride2",
        320, 320, 32, 64, 3, 3, 2, 2, 1, 1,
        3, 4, 7, 2, ACT_LEAKYRELU);

    // Pointwise 1x1 [1,160,160,64]→[1,160,160,32]
    fails += run_test(3, "pointwise 1x1",
        160, 160, 64, 32, 1, 1, 1, 1, 0, 0,
        3, 4, 7, 3, ACT_LEAKYRELU);

    // LCAM spatial attention 7x7 [1,160,160,2]→[1,160,160,1]
    fails += run_test(4, "LCAM spatial 7x7",
        160, 160, 2, 1, 7, 7, 1, 1, 3, 3,
        5, 4, 7, 7, ACT_NONE);

    // Small spatial [1,80,80,128]→[1,80,80,128]
    fails += run_test(5, "3x3 80x80 ch128",
        80, 80, 128, 128, 3, 3, 1, 1, 1, 1,
        4, 4, 7, 4, ACT_LEAKYRELU);

    // Head output 1x1 [1,20,20,128]→[1,20,20,4]
    fails += run_test(6, "head 1x1 20x20",
        20, 20, 128, 4, 1, 1, 1, 1, 0, 0,
        4, 4, 7, 5, ACT_NONE);

    printf("\n============================================================\n");
    printf(fails == 0 ? "ALL TESTS PASSED\n" : "SOME TESTS FAILED\n");
    printf("============================================================\n");
    return (fails == 0) ? 0 : 1;
}
