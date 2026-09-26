#include "head_transpose.h"
#include <cstdio>
#include <cstdlib>

// Software golden reference — NCHW to NHWC permutation
static void golden_transpose(
    int8_t* in, int8_t* out,
    int H, int W, int C)
{
    for (int c = 0; c < C; c++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                out[h*W*C + w*C + c] = in[c*H*W + h*W + w];
}

static int run_test(int id, int H, int W, int C)
{
    int total = H * W * C;
    int8_t* in      = new int8_t[total];
    int8_t* out_hls = new int8_t[total];
    int8_t* out_ref = new int8_t[total];

    // Fill with unique values so any incorrect index is immediately visible
    for (int i = 0; i < total; i++) {
        in[i] = (int8_t)((i % 200) - 100);
    }

    golden_transpose(in, out_ref, H, W, C);

    head_transpose((data_t*)in, (data_t*)out_hls, H, W, C);

    int mismatches = 0;
    for (int i = 0; i < total; i++) {
        if (out_hls[i] != out_ref[i]) {
            mismatches++;
            if (mismatches <= 5) {
                printf("  MISMATCH at dst_idx=%d: HLS=%d REF=%d\n",
                       i, (int)out_hls[i], (int)out_ref[i]);
            }
        }
    }

    printf("[Test %d] H=%d W=%d C=%d  [1,C,H,W]→[1,H,W,C]  -->  ",
           id, H, W, C);
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
    printf("head_transpose Testbench — 3 subgraph configurations\n");
    printf("============================================================\n\n");

    int fails = 0;
    // Subgraph [15]: [1,7,20,20] → [1,20,20,7]
    fails += run_test(0, 20, 20, 7);
    // Subgraph [18]: [1,7,40,40] → [1,40,40,7]
    fails += run_test(1, 40, 40, 7);
    // Subgraph [21]: [1,7,80,80] → [1,80,80,7]
    fails += run_test(2, 80, 80, 7);

    printf("\n============================================================\n");
    printf(fails == 0 ? "ALL TESTS PASSED\n" : "SOME TESTS FAILED\n");
    printf("============================================================\n");
    return (fails == 0) ? 0 : 1;
}
