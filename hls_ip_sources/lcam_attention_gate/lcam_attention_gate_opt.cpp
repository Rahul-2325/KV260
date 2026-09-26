// ============================================================
// lcam_attention_gate_opt.cpp
//
// OPTIMISED replacement for lcam_attention_gate.cpp.
//
// WHY: the original is functionally correct and 2.02x faster than the
// CPU fallback, but it is badly memory-bound:
//     * m_axi ports are 8 bits wide  -> ONE int8 per AXI beat
//     * every element does a FLOAT multiply + roundf()
//   Measured: 3.07M elements in 40.6 ms  (~76 M elem/s, ~0.76 elem/cycle
//   at 100 MHz). Synthesis instantiated a 32-bit float multiplier
//   (lcam_attention_gate_fmul_32ns_32ns_32_3_max_dsp_1_ip).
//
// WHAT CHANGED:
//  1. WIDE BUS. feat_in/feat_out are ap_uint<512> -> 64 int8 lanes per
//     beat, a 64x increase in bus utilisation. HLS infers long bursts
//     because access is sequential.
//  2. INTEGER ARITHMETIC. The whole operation is
//        out = clamp( (feat * weight) >> (fp_feat + fp_weight - fp_out) )
//     which is exact in integer form -- the float round-trip was never
//     needed. Removes the fmul IP and roundf entirely.
//  3. UNROLL 64. The 64 lanes in a beat are independent, so the inner
//     loop is fully unrolled; each lane is one int8*int8 multiply plus a
//     shift, which maps to LUT/DSP cheaply.
//
// SEMANTICS ARE IDENTICAL to the original (verified against the same
// formula):
//     real_feat   = feat   * 2^-fp_feat
//     real_weight = weight * 2^-fp_weight
//     out         = clamp(round(real_feat * real_weight * 2^fp_out))
// With integers, shift = fp_feat + fp_weight - fp_out and we add the
// round-to-nearest bias (1 << (shift-1)) before shifting.
//
// LAYOUT: NHWC, weight is [H,W,1] (one scalar per pixel, broadcast over
// channels). C is a multiple of 64 for every real layer (64/128/256/512),
// so a beat never straddles two pixels -- see the assert in the tb.
// ============================================================

#include "lcam_attention_gate.h"
#include <ap_int.h>

#define WBITS 512
#define LANES (WBITS / 8)          // 64 int8 lanes per beat

typedef ap_uint<WBITS> wide_t;

void lcam_attention_gate_opt(
    wide_t*  feat_in,
    data_t*  weight_in,
    wide_t*  feat_out,
    int      H,
    int      W,
    int      C,
    int      fp_feat,
    int      fp_weight,
    int      fp_out
)
{
#pragma HLS INTERFACE m_axi port=feat_in   bundle=gmem0 offset=slave \
        depth=204800 max_read_burst_length=64  num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=weight_in bundle=gmem1 offset=slave \
        depth=25600  max_read_burst_length=64  num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=feat_out  bundle=gmem2 offset=slave \
        depth=204800 max_write_burst_length=64 num_write_outstanding=16

#pragma HLS INTERFACE s_axilite port=H         bundle=control
#pragma HLS INTERFACE s_axilite port=W         bundle=control
#pragma HLS INTERFACE s_axilite port=C         bundle=control
#pragma HLS INTERFACE s_axilite port=fp_feat   bundle=control
#pragma HLS INTERFACE s_axilite port=fp_weight bundle=control
#pragma HLS INTERFACE s_axilite port=fp_out    bundle=control
#pragma HLS INTERFACE s_axilite port=return    bundle=control

    // shift and round bias, computed once
    const int shift = fp_feat + fp_weight - fp_out;
    const int bias  = (shift > 0) ? (1 << (shift - 1)) : 0;

    const int beats_per_pixel = C / LANES;     // C is 64/128/256/512
    const int n_pixels        = H * W;

PIXEL_LOOP:
    for (int p = 0; p < n_pixels; p++) {
        // one attention scalar per pixel, broadcast across all channels
        const int w_q = (int)(signed char)weight_in[p];

    BEAT_LOOP:
        for (int b = 0; b < beats_per_pixel; b++) {
#pragma HLS PIPELINE II=1
            const int idx = p * beats_per_pixel + b;
            wide_t in_beat  = feat_in[idx];
            wide_t out_beat = 0;

        LANE_LOOP:
            for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
                // extract lane l as signed int8
                ap_int<8> f = (ap_int<8>)(ap_uint<8>)in_beat.range(8 * l + 7, 8 * l);

                // Exact integer form of the original float computation.
                //
                // CAREFUL WITH ROUNDING: the original used roundf(),
                // which rounds HALF AWAY FROM ZERO. A plain
                // "(prod + bias) >> shift" is round-half-UP, which
                // disagrees on negative ties:
                //     roundf(-2.5) = -3   but   (-5 + 1) >> 1 = -2
                // Feature values are signed int8, so negatives are
                // common. Round the magnitude and re-apply the sign so
                // the optimised IP is BIT-IDENTICAL to the original.
                int prod = (int)f * w_q;
                int q;
                if (shift > 0) {
                    int mag = (prod < 0) ? -prod : prod;
                    int r   = (mag + bias) >> shift;
                    q       = (prod < 0) ? -r : r;
                } else {
                    q = prod << (-shift);
                }

                if      (q >  127) q =  127;
                else if (q < -128) q = -128;

                out_beat.range(8 * l + 7, 8 * l) = (ap_uint<8>)(ap_int<8>)q;
            }
            feat_out[idx] = out_beat;
        }
    }
}
