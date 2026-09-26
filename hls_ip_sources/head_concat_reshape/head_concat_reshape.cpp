#include "head_concat_reshape.h"

// ============================================================
// head_concat_reshape
//
// Covers subgraphs [22] and [23] in lcam_v5.xmodel.
//
// Pure DMA gather — three sequential memcpy-style bursts:
//   burst 1: copy head_20 (2800  bytes) to output[0]
//   burst 2: copy head_40 (11200 bytes) to output[2800]
//   burst 3: copy head_80 (44800 bytes) to output[14000]
//
// No arithmetic. The reshape is implicit — the three tensors
// are already in NHWC [H,W,7] order after head_transpose,
// so copying them sequentially gives [8400, 7] directly.
//
// Output feeds directly into your existing decode_0 HLS IP.
// ============================================================

void head_concat_reshape(
    data_t*  head_20,
    data_t*  head_40,
    data_t*  head_80,
    data_t*  output
)
{
    #pragma HLS INTERFACE m_axi port=head_20 bundle=gmem0 \
                                offset=slave depth=2800
    #pragma HLS INTERFACE m_axi port=head_40 bundle=gmem1 \
                                offset=slave depth=11200
    #pragma HLS INTERFACE m_axi port=head_80 bundle=gmem2 \
                                offset=slave depth=44800
    #pragma HLS INTERFACE m_axi port=output  bundle=gmem3 \
                                offset=slave depth=58800

    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Burst 1 — 20x20 head (400 anchors × 7 = 2800 elements)
    for (int i = 0; i < ANCHORS_20 * NUM_CLASSES; i++) {
        #pragma HLS PIPELINE II=1
        output[i] = head_20[i];
    }

    // Burst 2 — 40x40 head (1600 anchors × 7 = 11200 elements)
    int offset_40 = ANCHORS_20 * NUM_CLASSES;
    for (int i = 0; i < ANCHORS_40 * NUM_CLASSES; i++) {
        #pragma HLS PIPELINE II=1
        output[offset_40 + i] = head_40[i];
    }

    // Burst 3 — 80x80 head (6400 anchors × 7 = 44800 elements)
    int offset_80 = (ANCHORS_20 + ANCHORS_40) * NUM_CLASSES;
    for (int i = 0; i < ANCHORS_80 * NUM_CLASSES; i++) {
        #pragma HLS PIPELINE II=1
        output[offset_80 + i] = head_80[i];
    }
}
