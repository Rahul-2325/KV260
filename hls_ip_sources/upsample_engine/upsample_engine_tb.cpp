#include "upsample_engine.h"
#include <cstdio>

int main()
{
    printf("upsample_engine Testbench\n");
    printf("=========================\n\n");

    // Test: [1,20,20,256] → [1,40,40,256] scale=2
    int H=20, W=20, C=256, scale=2;
    int n_in  = H * W * C;
    int n_out = H*scale * W*scale * C;
    int8_t* in  = new int8_t[n_in];
    int8_t* out = new int8_t[n_out];
    int8_t* ref = new int8_t[n_out];

    for (int i = 0; i < n_in; i++)
        in[i] = (int8_t)((i % 100) - 50);

    // Golden: nearest neighbor
    for (int h = 0; h < H*scale; h++)
        for (int w = 0; w < W*scale; w++)
            for (int c = 0; c < C; c++)
                ref[(h*W*scale + w)*C + c] = in[((h/scale)*W + w/scale)*C + c];

    upsample_engine((data_t*)in, (data_t*)out, H, W, C, scale);

    int mismatches = 0;
    for (int i = 0; i < n_out; i++) {
        if (out[i] != ref[i]) {
            mismatches++;
            if (mismatches <= 3)
                printf("  MISMATCH[%d]: HLS=%d REF=%d\n",
                       i, (int)out[i], (int)ref[i]);
        }
    }

    printf("[%dx%dx%d → %dx%dx%d scale=%d]  -->  ",
           H, W, C, H*scale, W*scale, C, scale);
    printf(mismatches == 0 ? "PASS\n" : "FAIL (%d mismatches)\n", mismatches);

    delete[] in; delete[] out; delete[] ref;
    return (mismatches == 0) ? 0 : 1;
}
