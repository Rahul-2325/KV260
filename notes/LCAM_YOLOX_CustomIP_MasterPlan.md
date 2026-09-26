# LCAM-YOLOX Custom IP Master Plan
### Based on real xmodel subgraph analysis — lcam_v5.xmodel (Vitis-AI 3.0)
---

## 1. Complete Subgraph Map (24 total)

### 1.1 Full Graph Flow

```
[00] USER  (1 op)   → input node
[01] DPU   (58 ops) → backbone stem + Dark1-2, outputs [1,160,160,64] fp=4 + [1,160,160,1] fp=7
[02] CPU   (4 ops)  → LCAM2 Channel Attention gate apply
[03] DPU   (69 ops) → backbone Dark3, outputs [1,80,80,128] fp=5 + [1,80,80,1] fp=7
[04] CPU   (4 ops)  → LCAM2 Spatial Attention gate apply
[05] DPU   (69 ops) → backbone Dark4, outputs [1,40,40,256] fp=5 + [1,40,40,1] fp=7
[06] CPU   (4 ops)  → LCAM3 Channel Attention gate apply
[07] DPU   (64 ops) → backbone Dark5, outputs [1,20,20,512] fp=5 + [1,20,20,1] fp=7
[08] CPU   (4 ops)  → LCAM3 Spatial Attention gate apply
[09] DPU  (157 ops) → BaseConv + full FPN/PAN neck
[10] CPU   (3 ops)  → objectness sigmoid 20×20
[11] CPU   (3 ops)  → objectness sigmoid 40×40
[12] CPU   (3 ops)  → objectness sigmoid 80×80
[13] CPU   (3 ops)  → class sigmoid 20×20
[14] DPU   (8 ops)  → head conv 20×20, output [1,20,20,7] fp=5
[15] CPU   (3 ops)  → fix2float + transpose 20×20
[16] CPU   (3 ops)  → class sigmoid 40×40
[17] DPU   (8 ops)  → head conv 40×40, output [1,40,40,7] fp=5
[18] CPU   (3 ops)  → fix2float + transpose 40×40
[19] CPU   (3 ops)  → class sigmoid 80×80
[20] DPU   (8 ops)  → head conv 80×80, output [1,80,80,7] fp=5
[21] CPU   (3 ops)  → fix2float + transpose 80×80
[22] CPU   (5 ops)  → concat-fix + reshape-fix ×3
[23] CPU   (3 ops)  → final output fix
```

---

## 2. DPU Boundary Tensor Map (every handoff point)

| DPU Subgraph | Outputs to CPU | Shape | fix_point | Feeds into |
|---|---|---|---|---|
| [01] | feature map | [1,160,160,64] | fp=4 | CPU[02] channel attention |
| [01] | attention weights | [1,160,160,1] | fp=7 | CPU[02] mul gate |
| [03] | feature map | [1,80,80,128] | fp=5 | CPU[04] spatial attention |
| [03] | attention weights | [1,80,80,1] | fp=7 | CPU[04] mul gate |
| [05] | feature map | [1,40,40,256] | fp=5 | CPU[06] channel attention |
| [05] | attention weights | [1,40,40,1] | fp=7 | CPU[06] mul gate |
| [07] | feature map | [1,20,20,512] | fp=5 | CPU[08] spatial attention |
| [07] | attention weights | [1,20,20,1] | fp=7 | CPU[08] mul gate |
| [09] | bbox reg 40×40 | [1,40,40,2] | fp=2 | CPU[10-13] head sigmoids |
| [09] | bbox reg 80×80 | [1,80,80,2] | fp=2 | CPU head |
| [09] | feat 80×80 | [1,80,80,128] | fp=3 | DPU[20] head conv |
| [09] | bbox reg 20×20 | [1,20,20,2] | fp=3 | CPU head |
| [09] | obj 80×80 | [1,80,80,1] | fp=2 | CPU[12] sigmoid |
| [09] | obj 20×20 | [1,20,20,1] | fp=2 | CPU[10] sigmoid |
| [09] | feat 20×20 | [1,20,20,128] | fp=4 | DPU[14] head conv |
| [09] | feat 40×40 | [1,40,40,128] | fp=4 | DPU[17] head conv |
| [09] | obj 40×40 | [1,40,40,1] | fp=2 | CPU[11] sigmoid |
| [14] | head out 20×20 | [1,20,20,7] | fp=5 | CPU[15] transpose |
| [17] | head out 40×40 | [1,40,40,7] | fp=5 | CPU[18] transpose |
| [20] | head out 80×80 | [1,80,80,7] | fp=5 | CPU[21] transpose |

---

## 3. CPU Subgraph Detailed Spec

### Group A — LCAM Attention Gate Apply (subgraphs 02, 04, 06, 08)

These four are structurally identical — just different spatial sizes.

| Subgraph | Op sequence | Feature tensor | Attention tensor | Output | fix_point out |
|---|---|---|---|---|---|
| [02] | mul → fix2float → float2fix → fix2float | [1,160,160,64] fp=4 | [1,160,160,1] fp=7 | [1,160,160,64] | fp=5 |
| [04] | float2fix → mul → fix2float → fix2float | [1,80,80,128] fp=5 | [1,80,80,1] fp=7 | [1,80,80,128] | fp=5 |
| [06] | mul → fix2float → fix2float → float2fix | [1,40,40,256] fp=5 | [1,40,40,1] fp=7 | [1,40,40,256] | fp=5 |
| [08] | mul → fix2float → fix2float → float2fix | [1,20,20,512] fp=5 | [1,20,20,1] fp=7 | [1,20,20,512] | fp=6 |

**What the HLS IP does:**
- Input A: INT8 feature map (dequantize using fix_point_A)
- Input B: INT8 attention weight (dequantize using fix_point_B, broadcast across channels)
- Operation: elementwise multiply in float32
- Output: requantize result to INT8 at output fix_point
- Formula: `out_int8 = round(clip((A_float * B_float) * 2^fp_out, -128, 127))`

**One parameterizable HLS IP handles all four** — only H, W, C change.

---

### Group B — Head Objectness Sigmoid (subgraphs 10, 11, 12)

| Subgraph | Op sequence | Input shape | fix_point in | Output shape | fix_point out |
|---|---|---|---|---|---|
| [10] | float2fix → sigmoid → fix2float | [1,20,20,1] | fp=5 | [1,20,20,1] | float |
| [11] | sigmoid → float2fix → fix2float | [1,40,40,1] | fp=5 | [1,40,40,1] | float |
| [12] | sigmoid → fix2float → float2fix | [1,80,80,1] | fp=5 | [1,80,80,1] | float |

**What the HLS IP does:**
- Dequantize INT8 → float32: `x_f = x_int8 * 2^(-fix_point)`
- Apply sigmoid: `out = 1.0 / (1.0 + exp(-x_f))`
- Requantize: `out_int8 = round(clip(out * 2^fp_out, -128, 127))`
- Implement `exp()` using LUT or `hls::exp()` — fits in 2-3 DSPs with BRAM LUT

**One parameterizable IP handles all three** — only H, W differ.

---

### Group C — Head Class Sigmoid (subgraphs 13, 16, 19)

| Subgraph | Op sequence | Input shape | fix_point in | Output shape | fix_point out |
|---|---|---|---|---|---|
| [13] | float2fix → sigmoid → fix2float | [1,20,20,2] | fp=5 | [1,20,20,2] | float |
| [16] | sigmoid → fix2float → float2fix | [1,40,40,2] | fp=5 | [1,40,40,2] | float |
| [19] | sigmoid → fix2float → float2fix | [1,80,80,2] | fp=5 | [1,80,80,2] | float |

**Identical to Group B** — same IP, just C=2 instead of C=1.
Can be merged with Group B into a single sigmoid IP with C parameter.

---

### Group D — Head Transpose (subgraphs 15, 18, 21)

| Subgraph | Op sequence | Input shape | Output shape | Permutation |
|---|---|---|---|---|
| [15] | fix2float → fix2float → transpose | [1,7,20,20] | [1,7,20,20] | (0,2,3,1) → [1,20,20,7] |
| [18] | float2fix → fix2float → transpose | [1,7,40,40] | [1,7,40,40] | (0,2,3,1) → [1,40,40,7] |
| [21] | float2fix → fix2float → transpose | [1,7,80,80] | [1,7,80,80] | (0,2,3,1) → [1,80,80,7] |

**What the HLS IP does:**
- Pure data movement — no arithmetic
- Reads [N,C,H,W] layout, writes [N,H,W,C] layout
- For [1,7,H,W]: output[0][h][w][c] = input[0][c][h][w]
- Zero DSPs needed — pure BRAM/URAM addressing logic
- Can be implemented as address remapping with AXI master read/write

---

### Group E — Final Concat + Reshape (subgraph 22)

| Op | Input | Output | fix_point |
|---|---|---|---|
| concat-fix | 3× head outputs | [1,1,7,8400] | fp=5 |
| reshape-fix | [1,1,7,8400] | [1,7,8400] | fp=5 |
| reshape-fix | [1,7,8400] | [1,1,7,1600] | fp=5 |
| reshape-fix | [1,1,7,1600] | [1,1,7,400] | fp=5 |

8400 = 20×20 + 40×40 + 80×80 = 400 + 1600 + 6400 anchors total.

**What the HLS IP does:**
- Concatenate three INT8 tensors along anchor dimension in DDR
- Reshape is zero-copy — just metadata/stride change, no data movement needed
- Implementation: single DMA gather — read three buffers, write contiguous output

---

### Group F — Final Output Fix (subgraph 23)

3 ops, likely fix2float + scale + output sink. Minimal — feeds directly into decode_0 HLS IP which you already have working.

---

## 4. HLS IP Build Priority & Resource Estimates

| Priority | IP Name | Covers Subgraphs | DSPs | BRAMs | Latency est. | Why first |
|---|---|---|---|---|---|---|
| P1 | `lcam_attention_gate` | 02, 04, 06, 08 | ~4 | ~2 | <0.1ms | Largest feature tensors, on critical path between every DPU block |
| P2 | `head_sigmoid` | 10, 11, 12, 13, 16, 19 | ~6 | ~4 | <0.05ms | 6 subgraphs, one IP handles all — high reuse |
| P3 | `head_transpose` | 15, 18, 21 | 0 | ~8 | <0.05ms | Pure addressing — trivial logic, just needs AXI master |
| P4 | `head_concat_reshape` | 22 | 0 | ~4 | <0.1ms | Single DMA gather, no compute |
| P5 | `output_fix` | 23 | ~2 | ~1 | <0.01ms | Trivial scale op, feeds existing decode_0 |

**Total estimated additional resources:** ~12 DSPs, ~19 BRAMs on top of DPU.
Your available budget after DPU: ~387 BRAM, ~518 DSP remaining — well within limits.

---

## 5. Custom-Op Registration Strategy (Vitis-AI 3.0 / VART)

Each HLS IP plugs in via the Vitis-AI **custom CPU op** mechanism:

```
xmodel CPU subgraph
        ↓
   GraphRunner hits CPU subgraph
        ↓
   Calls registered C++ handler (your custom-op .so)
        ↓
   Handler: poke AXI-Lite regs → trigger HLS IP → DMA result back
        ↓
   Returns tensor to GraphRunner → next DPU subgraph starts
```

### Registration pattern (one per IP):

```cpp
// lcam_attention_gate_op.cpp
#include <vitis/ai/graph_runner.hpp>

class LCAMAttentionGateOp : public xir::Op {
public:
    void forward(
        const std::vector<vitis::ai::CpuFlatTensorBuffer*>& inputs,
        std::vector<vitis::ai::CpuFlatTensorBuffer*>& outputs) override
    {
        // 1. Get input pointers (feature map + attention weight)
        auto* feat   = inputs[0]->data<int8_t>();
        auto* weight = inputs[1]->data<int8_t>();
        auto* out    = outputs[0]->data<int8_t>();
        
        // 2. Get tensor shapes from buffer metadata
        auto shape = outputs[0]->get_tensor()->get_shape();
        int H = shape[1], W = shape[2], C = shape[3];
        
        // 3. Copy inputs to CMA buffers
        memcpy(cma_feat_buf, feat, H*W*C);
        memcpy(cma_weight_buf, weight, H*W*1);
        
        // 4. Set HLS IP registers via /dev/mem mmap
        write_reg(IP_BASE + CTRL_REG,   0x1);      // start
        write_reg(IP_BASE + H_REG,      H);
        write_reg(IP_BASE + W_REG,      W);
        write_reg(IP_BASE + C_REG,      C);
        write_reg(IP_BASE + FP_IN_REG,  fix_point_in);
        write_reg(IP_BASE + FP_OUT_REG, fix_point_out);
        
        // 5. Wait for ap_done
        while (!(read_reg(IP_BASE + CTRL_REG) & 0x2));
        
        // 6. Copy result back
        memcpy(out, cma_out_buf, H*W*C);
    }
};

// Register with VART
REGISTER_CUSTOM_OP("lcam_attention_gate", LCAMAttentionGateOp);
```

---

## 6. HLS IP Template — lcam_attention_gate

```cpp
// lcam_attention_gate.cpp
#include "ap_int.h"
#include "hls_stream.h"
#include <cmath>

// AXI-Lite control interface (same pattern as your decode_0/nms_top_0)
void lcam_attention_gate(
    ap_int<8>*  feat_in,       // [H*W*C] INT8 feature map
    ap_int<8>*  weight_in,     // [H*W*1] INT8 attention weights
    ap_int<8>*  feat_out,      // [H*W*C] INT8 output
    int         H,
    int         W,
    int         C,
    int         fp_feat,       // fix_point of feature tensor
    int         fp_weight,     // fix_point of attention tensor
    int         fp_out         // fix_point of output tensor
)
{
    #pragma HLS INTERFACE m_axi port=feat_in   bundle=gmem0 depth=1310720
    #pragma HLS INTERFACE m_axi port=weight_in bundle=gmem1 depth=20480
    #pragma HLS INTERFACE m_axi port=feat_out  bundle=gmem2 depth=1310720
    #pragma HLS INTERFACE s_axilite port=H
    #pragma HLS INTERFACE s_axilite port=W
    #pragma HLS INTERFACE s_axilite port=C
    #pragma HLS INTERFACE s_axilite port=fp_feat
    #pragma HLS INTERFACE s_axilite port=fp_weight
    #pragma HLS INTERFACE s_axilite port=fp_out
    #pragma HLS INTERFACE s_axilite port=return

    float scale_feat   = 1.0f / (1 << fp_feat);
    float scale_weight = 1.0f / (1 << fp_weight);
    float scale_out    = (float)(1 << fp_out);

    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            #pragma HLS PIPELINE II=1
            // Load attention weight (broadcast across C)
            float wt = (float)weight_in[h*W + w] * scale_weight;

            for (int c = 0; c < C; c++) {
                int idx = (h*W + w)*C + c;
                float f = (float)feat_in[idx] * scale_feat;
                float result = f * wt;
                // Requantize
                int q = (int)roundf(result * scale_out);
                q = q > 127 ? 127 : (q < -128 ? -128 : q);
                feat_out[idx] = (ap_int<8>)q;
            }
        }
    }
}
```

---

## 7. HLS IP Template — head_sigmoid

```cpp
void head_sigmoid(
    ap_int<8>*  data_in,   // [H*W*C] INT8
    ap_int<8>*  data_out,  // [H*W*C] INT8
    int         H, int W, int C,
    int         fp_in,
    int         fp_out
)
{
    #pragma HLS INTERFACE m_axi port=data_in  bundle=gmem0
    #pragma HLS INTERFACE m_axi port=data_out bundle=gmem1
    #pragma HLS INTERFACE s_axilite port=return

    float scale_in  = 1.0f / (1 << fp_in);
    float scale_out = (float)(1 << fp_out);

    for (int i = 0; i < H*W*C; i++) {
        #pragma HLS PIPELINE II=1
        float x = (float)data_in[i] * scale_in;
        float s = 1.0f / (1.0f + hls::exp(-x));
        int q = (int)roundf(s * scale_out);
        q = q > 127 ? 127 : (q < -128 ? -128 : q);
        data_out[i] = (ap_int<8>)q;
    }
}
```

---

## 8. HLS IP Template — head_transpose

```cpp
// [1,C,H,W] → [1,H,W,C] — NCHW to NHWC
void head_transpose(
    ap_int<8>*  data_in,   // [C*H*W] NCHW layout
    ap_int<8>*  data_out,  // [H*W*C] NHWC layout
    int H, int W, int C
)
{
    #pragma HLS INTERFACE m_axi port=data_in  bundle=gmem0
    #pragma HLS INTERFACE m_axi port=data_out bundle=gmem1
    #pragma HLS INTERFACE s_axilite port=return

    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                #pragma HLS PIPELINE II=1
                int src_idx = c*H*W + h*W + w;
                int dst_idx = h*W*C + w*C + c;
                data_out[dst_idx] = data_in[src_idx];
            }
        }
    }
}
```

---

## 9. Build Order & Integration Checklist

```
Phase 1 — HLS synthesis (Vitis HLS 2022.2)
  [ ] lcam_attention_gate  → csim → synth → export IP
  [ ] head_sigmoid         → csim → synth → export IP
  [ ] head_transpose       → csim → synth → export IP
  [ ] head_concat_reshape  → csim → synth → export IP

Phase 2 — Platform integration (Vivado 2022.2)
  [ ] Add all 4 IPs to kv260_lcam block design
  [ ] Assign AXI-Lite addresses (0x8000_0000 onward, after decode_0/nms_top_0)
  [ ] Connect AXI master ports to HP slave ports on PS
  [ ] Generate bitstream → new xclbin

Phase 3 — Custom-op registration (on KV260)
  [ ] Build custom-op .so for each IP
  [ ] Register op names matching xmodel CPU subgraph op types
  [ ] Test with GraphRunner: run single image, compare output vs ARM baseline
  [ ] Measure latency: target <1ms total for all 15 CPU subgraphs

Phase 4 — Full pipeline validation
  [ ] Run full inference loop (preprocess → xmodel → decode_0 → nms_top_0)
  [ ] Confirm mAP ≥ 78.11% (quantized baseline from paper)
  [ ] Measure end-to-end FPS improvement over ARM CPU subgraph baseline
```

---

## 10. Key Numbers to Keep Handy

| Parameter | Value |
|---|---|
| Board | KV260 (xck26-sfvc784-2LV-c) |
| DPU arch | DPUCZDX8G B4096, 1 core, 300 MHz |
| Available BRAM after DPU | ~387 |
| Available DSP after DPU | ~518 |
| Estimated IP DSP usage | ~12 |
| Estimated IP BRAM usage | ~19 |
| decode_0 latency (existing) | 0.27 ms |
| nms_top_0 latency (existing) | 0.12 ms |
| Target: all 15 CPU subgraphs | < 1.0 ms total |
| xmodel file | `/home/punnam_rahul/wildfire_project/compiled_v5/lcam_v5.xmodel` |
| Total subgraphs | 24 (8 DPU + 15 CPU + 1 USER) |
| Baseline FPS (paper) | 195 FPS (single DPU core) |
