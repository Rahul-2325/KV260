#include "head_concat_reshape.h"
#include <cstdio>
#include <cstring>

int main()
{
    printf("============================================================\n");
    printf("head_concat_reshape Testbench\n");
    printf("============================================================\n\n");

    // Allocate all buffers
    int8_t head_20[ANCHORS_20 * NUM_CLASSES];
    int8_t head_40[ANCHORS_40 * NUM_CLASSES];
    int8_t head_80[ANCHORS_80 * NUM_CLASSES];
    int8_t output [ANCHORS_TOTAL * NUM_CLASSES];
    int8_t output_ref[ANCHORS_TOTAL * NUM_CLASSES];

    // Fill each head with a distinct marker value so we can
    // verify exactly which source appears at each output position
    memset(head_20, 0x11, sizeof(head_20));   // 0x11 = 17
    memset(head_40, 0x22, sizeof(head_40));   // 0x22 = 34
    memset(head_80, 0x33, sizeof(head_80));   // 0x33 = 51

    // Build golden reference manually
    memcpy(output_ref,
           head_20, ANCHORS_20 * NUM_CLASSES);
    memcpy(output_ref + ANCHORS_20 * NUM_CLASSES,
           head_40, ANCHORS_40 * NUM_CLASSES);
    memcpy(output_ref + (ANCHORS_20 + ANCHORS_40) * NUM_CLASSES,
           head_80, ANCHORS_80 * NUM_CLASSES);

    // Call HLS DUT
    head_concat_reshape(
        (data_t*)head_20,
        (data_t*)head_40,
        (data_t*)head_80,
        (data_t*)output
    );

    // Compare
    int mismatches = 0;
    int total = ANCHORS_TOTAL * NUM_CLASSES;
    for (int i = 0; i < total; i++) {
        if (output[i] != output_ref[i]) {
            mismatches++;
            if (mismatches <= 5) {
                printf("  MISMATCH at idx=%d: HLS=%d REF=%d\n",
                       i, (int)output[i], (int)output_ref[i]);
            }
        }
    }

    // Spot-check boundaries
    printf("Boundary checks:\n");
    printf("  output[0]    = %d  (expect 17, from head_20)\n", (int)output[0]);
    printf("  output[%d]  = %d  (expect 34, first head_40 element)\n",
           ANCHORS_20*NUM_CLASSES, (int)output[ANCHORS_20*NUM_CLASSES]);
    printf("  output[%d] = %d  (expect 51, first head_80 element)\n",
           (ANCHORS_20+ANCHORS_40)*NUM_CLASSES,
           (int)output[(ANCHORS_20+ANCHORS_40)*NUM_CLASSES]);
    printf("  output[%d] = %d  (expect 51, last element)\n",
           total-1, (int)output[total-1]);

    printf("\n============================================================\n");
    if (mismatches == 0)
        printf("ALL TESTS PASSED  (%d elements verified)\n", total);
    else
        printf("FAILED: %d mismatches\n", mismatches);
    printf("============================================================\n");

    return (mismatches == 0) ? 0 : 1;
}
