#include "depthwise_engine.h"
#include <cstdio>
#include <cmath>

int main()
{
    printf("depthwise_engine Testbench\n");
    printf("==========================\n\n");

    int H=80, W=80, C=64, KH=3, KW=3;
    int stride_h=1, stride_w=1, pad_h=1, pad_w=1;
    int fp_in=4, fp_w=4, fp_bias=7, fp_out=4;
    int H_out = (H + 2*pad_h - KH) / stride_h + 1;
    int W_out = (W + 2*pad_w - KW) / stride_w + 1;
    int shift = fp_in + fp_w - fp_out;  // = 4

    int n_in  = H * W * C;
    int n_w   = C * KH * KW;
    int n_out = H_out * W_out * C;

    int8_t* in  = new int8_t[n_in];
    int8_t* wt  = new int8_t[n_w];
    int8_t* b   = new int8_t[C];
    int8_t* out = new int8_t[n_out];
    int8_t* ref = new int8_t[n_out];

    // Fill inputs
    for (int i = 0; i < n_in; i++) in[i] = (int8_t)((i % 60) - 30);
    for (int i = 0; i < n_w;  i++) wt[i] = (int8_t)((i % 20) - 10);
    // Bias = 0 to isolate conv math first
    for (int i = 0; i < C; i++) b[i] = 0;

    // Golden reference
    for (int h = 0; h < H_out; h++) {
        for (int w2 = 0; w2 < W_out; w2++) {
            for (int c = 0; c < C; c++) {

                int32_t acc = 0;
                for (int kh = 0; kh < KH; kh++) {
                    for (int kw = 0; kw < KW; kw++) {
                        int ih = h * stride_h - pad_h + kh;
                        int iw = w2 * stride_w - pad_w + kw;
                        if (ih < 0 || ih >= H || iw < 0 || iw >= W)
                            continue;
                        int fi = (ih * W + iw) * C + c;
                        int wi = (c * KH + kh) * KW + kw;
                        acc += (int32_t)in[fi] * (int32_t)wt[wi];
                    }
                }

                // No bias (b=0)
                int out_int;
                if (shift >= 0)
                    out_int = (int)(acc >> shift);
                else
                    out_int = (int)(acc << (-shift));

                // LeakyReLU
                if (out_int < 0) out_int = out_int / 10;

                // Saturate
                if      (out_int >  127) out_int =  127;
                else if (out_int < -128) out_int = -128;

                ref[(h * W_out + w2) * C + c] = (int8_t)out_int;
            }
        }
    }

    // Call HLS DUT
    depthwise_engine(
        (data_t*)in, (data_t*)wt, (data_t*)b, (data_t*)out,
        H, W, C, KH, KW,
        stride_h, stride_w, pad_h, pad_w,
        fp_in, fp_w, fp_bias, fp_out
    );

    // Compare
    int mismatches = 0;
    for (int i = 0; i < n_out; i++) {
        if (abs((int)out[i] - (int)ref[i]) > 1) {
            mismatches++;
            if (mismatches <= 5)
                printf("  MISMATCH[%d]: HLS=%d REF=%d\n",
                       i, (int)out[i], (int)ref[i]);
        }
    }

    printf("[%dx%dx%d DW %dx%d bias=0]  -->  ", H, W, C, KH, KW);
    if (mismatches == 0)
        printf("PASS (%d elements)\n", n_out);
    else
        printf("FAIL (%d mismatches)\n", mismatches);

    delete[] in; delete[] wt; delete[] b;
    delete[] out; delete[] ref;
    return (mismatches == 0) ? 0 : 1;
}
