#include "head_sigmoid.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Software golden reference
static void golden_sigmoid(
    int8_t* in, int8_t* out,
    int H, int W, int C,
    int fp_in, int fp_out)
{
    float scale_in  = 1.0f / (float)(1 << fp_in);
    float scale_out = (float)(1 << fp_out);
    int total = H * W * C;
    for (int i = 0; i < total; i++) {
        float x = (float)in[i] * scale_in;
        float s = 1.0f / (1.0f + expf(-x));
        int q = (int)roundf(s * scale_out);
        if      (q >  127) q =  127;
        else if (q < -128) q = -128;
        out[i] = (int8_t)q;
    }
}

static int run_test(int id, int H, int W, int C, int fp_in, int fp_out)
{
    int total = H * W * C;
    int8_t* in      = new int8_t[total];
    int8_t* out_hls = new int8_t[total];
    int8_t* out_ref = new int8_t[total];

    // Fill with full INT8 range — exercises all sigmoid input regions
    for (int i = 0; i < total; i++) {
        in[i] = (int8_t)((i % 256) - 128);
    }

    golden_sigmoid(in, out_ref, H, W, C, fp_in, fp_out);

    head_sigmoid(
        (data_t*)in, (data_t*)out_hls,
        H, W, C, fp_in, fp_out
    );

    int mismatches = 0;
    for (int i = 0; i < total; i++) {
        // Allow ±1 difference due to float rounding between
        // software expf() and hardware hls::exp() approximation
        if (abs((int)out_hls[i] - (int)out_ref[i]) > 1) {
            mismatches++;
            if (mismatches <= 5) {
                printf("  MISMATCH at idx=%d: HLS=%d REF=%d input=%d\n",
                       i, (int)out_hls[i], (int)out_ref[i], (int)in[i]);
            }
        }
    }

    printf("[Test %d] H=%d W=%d C=%d fp_in=%d fp_out=%d  -->  ",
           id, H, W, C, fp_in, fp_out);
    if (mismatches == 0)
        printf("PASS (%d elements)\n", total);
    else
        printf("FAIL (%d mismatches)\n", mismatches);

    delete[] in;
    delete[] out_hls;
    delete[] out_ref;
    return mismatches;
}

int main()
{
    printf("============================================================\n");
    printf("head_sigmoid Testbench — 6 subgraph configurations\n");
    printf("============================================================\n\n");

    int fails = 0;

    // Subgraph [10] — objectness 20x20
    fails += run_test(0, 20, 20, 1, 5, 5);
    // Subgraph [11] — objectness 40x40
    fails += run_test(1, 40, 40, 1, 5, 5);
    // Subgraph [12] — objectness 80x80
    fails += run_test(2, 80, 80, 1, 5, 5);
    // Subgraph [13] — class 20x20
    fails += run_test(3, 20, 20, 2, 5, 5);
    // Subgraph [16] — class 40x40
    fails += run_test(4, 40, 40, 2, 5, 5);
    // Subgraph [19] — class 80x80
    fails += run_test(5, 80, 80, 2, 5, 5);

    printf("\n============================================================\n");
    printf(fails == 0 ? "ALL TESTS PASSED\n" : "SOME TESTS FAILED\n");
    printf("============================================================\n");
    return (fails == 0) ? 0 : 1;
}
