#include "lcam_attention_gate.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ============================================================
// TESTBENCH for lcam_attention_gate
//
// Tests all 4 real configurations from lcam_v5.xmodel:
//   Config 0: H=160,W=160,C=64  fp_feat=4,fp_weight=7,fp_out=5  [sg02]
//   Config 1: H=80, W=80, C=128 fp_feat=5,fp_weight=7,fp_out=5  [sg04]
//   Config 2: H=40, W=40, C=256 fp_feat=5,fp_weight=7,fp_out=5  [sg06]
//   Config 3: H=20, W=20, C=512 fp_feat=5,fp_weight=7,fp_out=6  [sg08]
//
// For each config:
//   1. Fill input tensors with known values
//   2. Compute expected output in software (golden reference)
//   3. Call the HLS DUT (Device Under Test)
//   4. Compare output vs golden — report PASS/FAIL
// ============================================================

// Software golden reference — exact same math as the HLS IP
// so we know what the correct answer is before synthesis
static void golden_attention_gate(
    int8_t* feat_in,
    int8_t* weight_in,
    int8_t* feat_out_golden,
    int H, int W, int C,
    int fp_feat, int fp_weight, int fp_out)
{
    float scale_feat   = 1.0f / (float)(1 << fp_feat);
    float scale_weight = 1.0f / (float)(1 << fp_weight);
    float scale_out    = (float)(1 << fp_out);

    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            float wt = (float)weight_in[h*W + w] * scale_weight;
            for (int c = 0; c < C; c++) {
                int idx      = (h*W + w)*C + c;
                float f      = (float)feat_in[idx] * scale_feat;
                float result = f * wt;
                int   q      = (int)roundf(result * scale_out);
                if      (q >  127) q =  127;
                else if (q < -128) q = -128;
                feat_out_golden[idx] = (int8_t)q;
            }
        }
    }
}

// Run one test configuration, return number of mismatches
static int run_test(
    int test_id,
    int H, int W, int C,
    int fp_feat, int fp_weight, int fp_out)
{
    int total_feat   = H * W * C;
    int total_weight = H * W;

    // Allocate buffers
    int8_t* feat_in        = new int8_t[total_feat];
    int8_t* weight_in      = new int8_t[total_weight];
    int8_t* feat_out_hls   = new int8_t[total_feat];
    int8_t* feat_out_golden= new int8_t[total_feat];

    // Fill feat_in with a repeating ramp pattern
    // Range [-64, 63] to stay well within INT8 and exercise both signs
    for (int i = 0; i < total_feat; i++) {
        feat_in[i] = (int8_t)((i % 128) - 64);
    }

    // Fill weight_in with moderate positive values
    // Simulates typical sigmoid attention weights (post-quantization)
    // Real weights after sigmoid are [0,1], quantized at fp=7 they are [0, 127]
    for (int i = 0; i < total_weight; i++) {
        weight_in[i] = (int8_t)((i % 100) + 10);   // range [10, 109]
    }

    // Compute golden reference
    golden_attention_gate(feat_in, weight_in, feat_out_golden,
                          H, W, C, fp_feat, fp_weight, fp_out);

    // Call HLS DUT
    lcam_attention_gate(
        (data_t*)feat_in,
        (data_t*)weight_in,
        (data_t*)feat_out_hls,
        H, W, C,
        fp_feat, fp_weight, fp_out
    );

    // Compare outputs
    int mismatches = 0;
    for (int i = 0; i < total_feat; i++) {
        if (feat_out_hls[i] != feat_out_golden[i]) {
            mismatches++;
            if (mismatches <= 5) {   // print first 5 mismatches only
                printf("  MISMATCH at idx=%d: HLS=%d  GOLDEN=%d\n",
                       i, (int)feat_out_hls[i], (int)feat_out_golden[i]);
            }
        }
    }

    printf("[Test %d] H=%d W=%d C=%d fp_feat=%d fp_weight=%d fp_out=%d  -->  ",
           test_id, H, W, C, fp_feat, fp_weight, fp_out);

    if (mismatches == 0) {
        printf("PASS  (%d elements checked)\n", total_feat);
    } else {
        printf("FAIL  (%d/%d mismatches)\n", mismatches, total_feat);
    }

    // Cleanup
    delete[] feat_in;
    delete[] weight_in;
    delete[] feat_out_hls;
    delete[] feat_out_golden;

    return mismatches;
}

int main()
{
    printf("============================================================\n");
    printf("lcam_attention_gate C-Simulation Testbench\n");
    printf("Testing all 4 real subgraph configurations from lcam_v5.xmodel\n");
    printf("============================================================\n\n");

    int total_fails = 0;

    // Config 0 — Subgraph [02]: largest tensor, LCAM2 channel attention
    total_fails += run_test(0,
        /*H=*/160, /*W=*/160, /*C=*/64,
        /*fp_feat=*/4, /*fp_weight=*/7, /*fp_out=*/5);

    // Config 1 — Subgraph [04]: LCAM2 spatial attention
    total_fails += run_test(1,
        /*H=*/80, /*W=*/80, /*C=*/128,
        /*fp_feat=*/5, /*fp_weight=*/7, /*fp_out=*/5);

    // Config 2 — Subgraph [06]: LCAM3 channel attention
    total_fails += run_test(2,
        /*H=*/40, /*W=*/40, /*C=*/256,
        /*fp_feat=*/5, /*fp_weight=*/7, /*fp_out=*/5);

    // Config 3 — Subgraph [08]: LCAM3 spatial attention, smallest tensor
    total_fails += run_test(3,
        /*H=*/20, /*W=*/20, /*C=*/512,
        /*fp_feat=*/5, /*fp_weight=*/7, /*fp_out=*/6);

    printf("\n============================================================\n");
    if (total_fails == 0) {
        printf("ALL TESTS PASSED\n");
        printf("Ready to proceed to C/RTL co-simulation and synthesis.\n");
    } else {
        printf("FAILED: %d test(s) had mismatches. Fix before synthesizing.\n",
               total_fails);
    }
    printf("============================================================\n");

    return (total_fails == 0) ? 0 : 1;
}
