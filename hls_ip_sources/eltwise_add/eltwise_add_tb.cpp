#include "eltwise_add.h"
#include <cstdio>
#include <cmath>

int main()
{
    printf("eltwise_add Testbench\n");
    printf("====================\n\n");

    // Test: [1,160,160,64] fp_a=3 fp_b=3 fp_out=3
    int H=160, W=160, C=64;
    int total = H*W*C;
    int8_t* a   = new int8_t[total];
    int8_t* b   = new int8_t[total];
    int8_t* out = new int8_t[total];
    int8_t* ref = new int8_t[total];

    for (int i = 0; i < total; i++) {
        a[i] = (int8_t)((i % 80) - 40);
        b[i] = (int8_t)((i % 60) - 30);
    }

    // Golden
    for (int i = 0; i < total; i++) {
        int r = (int)a[i] + (int)b[i];  // same fp, no shift needed
        if      (r >  127) r =  127;
        else if (r < -128) r = -128;
        ref[i] = (int8_t)r;
    }

    eltwise_add((data_t*)a, (data_t*)b, (data_t*)out,
                H, W, C, 3, 3, 3);

    int mismatches = 0;
    for (int i = 0; i < total; i++) {
        if (abs((int)out[i] - (int)ref[i]) > 1) {
            mismatches++;
            if (mismatches <= 3)
                printf("  MISMATCH[%d]: HLS=%d REF=%d\n",
                       i, (int)out[i], (int)ref[i]);
        }
    }

    printf("H=%d W=%d C=%d fp=3,3→3  -->  ", H, W, C);
    if (mismatches == 0)
        printf("PASS (%d elements)\n", total);
    else
        printf("FAIL (%d mismatches)\n", mismatches);

    delete[] a; delete[] b; delete[] out; delete[] ref;
    return (mismatches == 0) ? 0 : 1;
}
