#include "pool_engine.h"
#include <cstdio>
#include <cmath>

int main()
{
    printf("pool_engine Testbench\n");
    printf("====================\n\n");

    // Test: [1,160,160,64] → [1,1,1,64]
    int H=160, W=160, C=64;
    int total = H*W*C;
    int8_t* in  = new int8_t[total];
    int8_t* out = new int8_t[C];
    int8_t* ref = new int8_t[C];

    for (int i = 0; i < total; i++)
        in[i] = (int8_t)((i % 80) - 40);

    // Golden
    for (int c = 0; c < C; c++) {
        int sum = 0;
        for (int i = 0; i < H*W; i++)
            sum += (int)in[i*C + c];
        int avg = sum / (H*W);
        if      (avg >  127) avg =  127;
        else if (avg < -128) avg = -128;
        ref[c] = (int8_t)avg;
    }

    pool_engine((data_t*)in, (data_t*)out, H, W, C, 4, 4);

    int mismatches = 0;
    for (int i = 0; i < C; i++) {
        if (abs((int)out[i] - (int)ref[i]) > 1) {
            mismatches++;
            if (mismatches <= 3)
                printf("  MISMATCH[%d]: HLS=%d REF=%d\n",
                       i, (int)out[i], (int)ref[i]);
        }
    }

    printf("[%dx%dx%d → 1x1x%d]  -->  ", H, W, C, C);
    printf(mismatches == 0 ? "PASS\n" : "FAIL (%d mismatches)\n", mismatches);

    delete[] in; delete[] out; delete[] ref;
    return (mismatches == 0) ? 0 : 1;
}
