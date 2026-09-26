#ifndef UPSAMPLE_ENGINE_H
#define UPSAMPLE_ENGINE_H
#include "ap_int.h"
typedef ap_int<8> data_t;
#define MAX_UPSAMPLE (40 * 40 * 512)

// Nearest-neighbor upsample — used in FPN/PAN neck
// scale_factor is always 2 in YOLOX FPN
void upsample_engine(
    data_t* feat_in,   // [H * W * C]
    data_t* feat_out,  // [H*scale * W*scale * C]
    int     H,
    int     W,
    int     C,
    int     scale      // always 2 in this model
);
#endif
