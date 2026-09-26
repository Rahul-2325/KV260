#ifndef LCAM_ATTENTION_GATE_H
#define LCAM_ATTENTION_GATE_H

#include "ap_int.h"

// ============================================================
// Maximum tensor dimensions (for array sizing in HLS)
// Largest case: subgraph [02] → [1, 160, 160, 64]
// ============================================================
#define MAX_H       160
#define MAX_W       160
#define MAX_C       512
#define MAX_HW      (MAX_H * MAX_W)          // 25600
#define MAX_HWC     (MAX_H * MAX_W * MAX_C)  // 13107200

// ============================================================
// Data types
// ============================================================
typedef ap_int<8>   data_t;    // INT8 — all tensors are INT8
typedef ap_int<32>  ctrl_t;    // control registers

// ============================================================
// Top-level function declaration
// This is what Vitis HLS synthesizes into RTL
// ============================================================
void lcam_attention_gate(
    data_t*  feat_in,      // [H * W * C]  INT8 feature map from DPU
    data_t*  weight_in,    // [H * W * 1]  INT8 attention weights from DPU
    data_t*  feat_out,     // [H * W * C]  INT8 gated output → back to DPU
    int      H,            // spatial height  (160 / 80 / 40 / 20)
    int      W,            // spatial width   (160 / 80 / 40 / 20)
    int      C,            // channels        (64 / 128 / 256 / 512)
    int      fp_feat,      // fix_point of feat_in  (4 or 5)
    int      fp_weight,    // fix_point of weight_in (7 always)
    int      fp_out        // fix_point of feat_out  (5 or 6)
);

#endif // LCAM_ATTENTION_GATE_H
