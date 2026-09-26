#include "head_transpose.h"

void head_transpose(
    data_t* data_in,
    data_t* data_out,
    int H,
    int W,
    int C)
{
    #pragma HLS INTERFACE m_axi port=data_in  offset=slave bundle=gmem0 depth=76800
    #pragma HLS INTERFACE m_axi port=data_out offset=slave bundle=gmem1 depth=76800
    #pragma HLS INTERFACE s_axilite port=data_in  bundle=control
    #pragma HLS INTERFACE s_axilite port=data_out bundle=control
    #pragma HLS INTERFACE s_axilite port=H bundle=control
    #pragma HLS INTERFACE s_axilite port=W bundle=control
    #pragma HLS INTERFACE s_axilite port=C bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                #pragma HLS PIPELINE II=1
                data_out[(h*W + w)*C + c] = data_in[(c*H + h)*W + w];
            }
        }
    }
}
