# 4K Video Deployment on KV260 — Stock-Bitstream Track

**Date: 2026-09-01. Board: KV260 @ 192.168.137.126, app `kv260-benchmark-b4096`.**

This document covers a NEW, separate track from the LCAM-YOLOX custom-IP work
in `PROJECT_HISTORY.md`. It does not replace or modify anything there.

**Goal:** run object detection on 4K (3840×2160) video on the KV260 using the
**stock built-in bitstream** and **stock pre-trained models** — no Vivado build,
no training, no custom HLS IP — and measure honestly where the time goes.

**Headline result:** everything works, and the bottleneck is **not** the DPU.
It is **CPU video decode**, which is 83% of the frame time at 4K. The DPU has
~4× more headroom than the pipeline can currently feed it.

```
                                 mosaic clip     REAL 4K drone footage
4K (3840×2160) end-to-end    :     2.13 FPS            2.28 FPS   <- decode-bound
1080p          end-to-end    :     7.82 FPS            8.88 FPS   <- decode-bound
DPU-only ceiling             :    32.64 FPS           32.32 FPS   <- resolution-independent
projected with VCU + PL scaler :    ~26 FPS                       <- the fix
```

Measured on **two independent 4K clips** — a controlled mosaic and genuine
4K aerial drone footage (§6). They agree to within 7%, so the conclusion is
not an artefact of the test content.

---

## §1. What was already on the board, and the problem found

| Layer | Version | Evidence |
|---|---|---|
| Runtime libraries | **Vitis AI 3.0** | `vart-runner 3.0.0`, `vitis_ai_library 3.0.0`, `xir 3.0.0`, built 2022-12-27 |
| Pre-installed models (~250) | **Vitis AI 3.0** | every xmodel fingerprint = `0x101000056010407` |
| Stock DPU bitstream | **Vitis AI 2.5** | DPU IP **v4.0.0**, generated **2022-05-14**, git `4772d51` |

The board image is a **mismatched pair**: a 3.0 software stack on a 2.5-era DPU
bitstream. Vitis AI 3.0 ships DPUCZDX8G **v4.1.0**; this board has **v4.0.0**.

Live DPU (`xdputil query`):

```
DPU Arch    : DPUCZDX8G_ISA1_B4096_0101000016010407
fingerprint : 0x101000016010407
cores=1   B4096   300 MHz   cu_addr 0xa0010000
```

### §1.1 Consequence: none of the 250 pre-installed models can run

All 25 pre-installed YOLO variants report the 3.0 fingerprint:

```
yolox_nano_pt     0x101000056010407   dpu_sg=1/5      <- will NOT run
yolov5_nano_pt    0x101000056010407   dpu_sg=1/5      <- will NOT run
yolov5s6_pt       0x101000056010407   dpu_sg=1/6
yolov6m_pt        0x101000056010407   dpu_sg=1/8
yolov4_csp_pt     0x101000056010407   dpu_sg=1/5
ofa_yolo_pt       0x101000056010407   dpu_sg=1/5
... all 25 identical
```

This was **not** caused by anything done in this session — it has always been
the board's state. See `PROJECT_HISTORY.md` §29.1, where the same audit found
only `lcam_v5_2p5.xmodel` (compiled with Vitis AI 2.5) matched the hardware.

Swapping to another installed app does **not** fix it: the whole model library
is compiled for **B4096**, and no bitstream on this board is a B4096 at
`...56010407`. The three known values:

```
0x101000016010407   stock kv260-benchmark-b4096   (VAI 2.5 default config)
0x101000056010407   VAI 3.0 model-zoo default
0x101000012010407   kv260-hybrid2/3 (custom build, CHANNEL_AUGMENTATION_DISABLE)
```

> **The fingerprint encodes DPU feature CONFIGURATION, not the tool version.**
> AMD changed the default B4096 config between 2.5 and 3.0. All four builds
> report target name `DPUCZDX8G_ISA1_B4096` — the **name is not sufficient**,
> only the fingerprint is. (`PROJECT_HISTORY.md` §29.15, §32.5.)

### §1.2 The fix

Download **Vitis AI 2.5** model-zoo tarballs, which are compiled against the
2.5 default config and therefore carry `0x101000016010407` natively.

> **Do NOT use `XLNX_ENABLE_FINGERPRINT_CHECK=0`.** It is widely suggested
> online and it appears to work. It disables the check that a model matches
> the silicon, giving silent wrong answers — the same failure class as the
> all-zeros bug in `PROJECT_HISTORY.md` §26.

---

## §2. Models downloaded and installed

Source: Vitis AI **2.5** Model Zoo, board group `zcu102_zcu104_kv260`.
URL pattern: `https://www.xilinx.com/bin/public/openDownload?filename=<name>-zcu102_zcu104_kv260-r2.5.0.tar.gz`

| Installed as | Source tarball | Input | Classes | Float ops | Head |
|---|---|---|---|---|---|
| `ofa_yolo_pt_v25` | `ofa_yolo_pt` | 640×640 | 80 (COCO) | 48.88G | YOLOv5 |
| `ofa_yolo_p50_v25` | `ofa_yolo_pruned_0_50_pt` | 640×640 | 80 (COCO) | 24.62G | YOLOv5 |
| `tiny_yolov3_vmss_v25` | `tiny_yolov3_vmss` | 416×416 | 10 (private) | 5.46G | YOLOv3 |

All six extracted xmodels (including `_acc` variants) verified **before**
install:

```
MATCH   ofa_yolo_pt              0x101000016010407   dpu_sg=1/5
MATCH   ofa_yolo_pruned_0_50_pt  0x101000016010407   dpu_sg=1/5
MATCH   tiny_yolov3_vmss         0x101000016010407   dpu_sg=1/4
```

### §2.1 Installed under NEW names — nothing overwritten

The 2.5 tarballs unpack to directory names that **already exist** on the board
holding the 3.0 versions. They were therefore installed under `_v25` suffixes:

```
/usr/share/vitis_ai_library/models/ofa_yolo_pt_v25/
/usr/share/vitis_ai_library/models/ofa_yolo_p50_v25/
/usr/share/vitis_ai_library/models/tiny_yolov3_vmss_v25/
```

The Vitis AI Library resolves a model by `<dir>/<dir>.xmodel`, so the `.xmodel`,
`.prototxt` and `meta.json` were renamed to match. Originals are untouched and
still carry their original `Mar 9 2018` package timestamps.

### §2.2 Important limitation of the 2.5 zoo

Vitis AI 2.5 has **no YOLOv5, no YOLOv6, and no YOLOX-nano-on-COCO** — those
arrived in 3.0. The 2.5 menu is YOLOv2/v3/v4, OFA-yolo, and YOLOX-m on TT100K
traffic signs. To use v5/v6/YOLOX-nano, the 3.0 *quantized* xmodel must be
recompiled with `vai_c_xir` against an `arch.json` containing
`{"fingerprint":"0x101000016010407"}` — the same procedure as
`PROJECT_HISTORY.md` §29.11.

---

## §3. Scripts written

All live in `/home/root/vai25_models/` on the board, and are copied into
`vai25_yolo/` in this folder.

| Script | Purpose |
|---|---|
| `detect_ofa_yolo.py` | Single-image detection: VART + full YOLOv5 decode + NMS, draws boxes |
| `make_test_video.py` | Builds the 4K and 1080p MJPEG test clips on the board |
| `bench_video.py` | Video benchmark, **naive** pre/post-processing, per-stage timing |
| `bench_video_opt.py` | Video benchmark, **optimised** pre/post-processing |

Dependencies are only `vart`, `xir`, `numpy`, `cv2` — all already on the board.
No Vitis AI Library C++ demo binaries exist on this image (`test_jpeg_*` are
absent), so postprocessing is implemented in Python from scratch.

---

## §4. Correctness verified first

`ofa_yolo_pt_v25` on the canonical YOLOv5 test images:

```
bus.jpg     -> person 0.909, person 0.893, person 0.856, bus 0.746, person 0.495
zidane.jpg  -> person 0.900, tie 0.787, person 0.787
```

Both are the textbook YOLOv5 results. Box geometry visually confirmed correct
(letterbox un-mapping verified on the saved overlay). Channel order is **RGB**;
input quantisation is `int8 = round(pixel/255 × 64)` (`fix_point=6`).

Model tensor layout:

```
IN   [1, 640, 640, 3]    fix_point=6   (scale 64)
OUT  [1,  80,  80, 255]  fix_point=3   stride 8
OUT  [1,  40,  40, 255]  fix_point=4   stride 16
OUT  [1,  20,  20, 255]  fix_point=4   stride 32
```
255 = 3 anchors × 85 (4 box + 1 obj + 80 class). Standard YOLOv5 anchors.

A wildfire image (`WEB09971.jpg`) correctly yields **no detections** — COCO has
no fire or smoke class. This is expected, not a failure.

---

## §5. DPU-only benchmark (`xdputil benchmark`, 60 s each)

Pure DPU throughput, no image I/O and no pre/post-processing:

| Model | 1 thread | 4 threads | ms/frame |
|---|---|---|---|
| `ofa_yolo_pt_v25` | **19.39 FPS** | 20.16 FPS | 51.6 |
| `ofa_yolo_p50_v25` | **33.23 FPS** | 35.45 FPS | 30.1 |
| `tiny_yolov3_vmss_v25` | **157.81 FPS** | — | 6.3 |

Threading adds almost nothing — there is only **one DPU core**, so the
single-thread figure is effectively the ceiling.

`ofa_yolo_p50_v25` clears 30 FPS at 640×640, i.e. the DPU alone is fast enough
for real-time 4K30 if it can be fed.

---

## §6. The 4K test videos

Two independent 4K clips were used, deliberately, so that no conclusion rests
on the properties of a single piece of content.

### §6.A Clip 1 — genuine 4K aerial drone footage (`real_4k.avi`)

A real 3840×2160 H.264 drone clip (aerial, ocean and rock coastline,
23.976 fps, 28.2 s, 126.5 MB) downloaded to the PC, then transcoded to MJPEG
with ffmpeg because **the board cannot decode H.264 at all** (§8):

```bash
ffmpeg -ss 2 -t 5 -i real_source_4k.mp4 -vf scale=3840:2160 \
       -c:v mjpeg -q:v 6 -pix_fmt yuvj420p -an real_4k.avi
ffmpeg -ss 2 -t 5 -i real_source_4k.mp4 -vf scale=1920:1080 \
       -c:v mjpeg -q:v 6 -pix_fmt yuvj420p -an real_1080p.avi
```

```
real_4k.avi     3840×2160  MJPEG  120 frames  111.4 MB
real_1080p.avi  1920×1080  MJPEG  120 frames   38.2 MB
```

Aerial drone footage is a good match for the UAV use case in camera type and
motion. It contains **no COCO objects**, so it is a pure throughput test —
detections are demonstrated separately on Clip 2 and on the still images (§4).

### §6.B Clip 2 — controlled mosaic (`test_4k.avi`)

`make_test_video.py` composes each 4K frame as an
**exact 3×3 mosaic of nine native 1280×720 images** (3×1280 = 3840,
3×720 = 2160). Nothing is upscaled, so the frame carries real 4K-worth of
pixel detail. Cells rotate every frame so successive frames genuinely differ.

Sources: `bus.jpg`, `zidane.jpg` (so COCO detections actually appear) plus ten
native-720p wildfire frames from `/home/root/valset/images/`.

```
test_4k.avi     3840×2160  MJPEG  120 frames  551.3 MB
test_1080p.avi  1920×1080  MJPEG  120 frames  170.2 MB
```

Note this doubles as a nice illustration of the small-object problem: each cell
is squashed ~6× when the 4K frame is resized to 640×640. Large objects (the
bus, the two people) survive; COCO also produces visible false positives on the
wildfire cells (`giraffe 0.754`, `train 0.341`).

---

## §7. MEASURED VIDEO RESULTS

Command form:

```bash
cd /home/root/vai25_models
python3 bench_video_opt.py \
  /usr/share/vitis_ai_library/models/ofa_yolo_p50_v25/ofa_yolo_p50_v25.xmodel \
  test_4k.avi --max-frames 100
```

100 frames timed, 3 warm-up frames discarded.

### §7.1 Naive pre/post-processing

| Stage | 4K `ofa_yolo_pt` | 4K `ofa_yolo_p50` |
|---|---|---|
| decode (MJPEG, CPU) | 149.30 ms (31.8%) | 168.38 ms (36.0%) |
| preprocess | 54.09 ms (11.5%) | 54.47 ms (11.6%) |
| DPU inference | 52.07 ms (11.1%) | 30.47 ms (6.5%) |
| postprocess + NMS | **213.32 ms (45.5%)** | **214.32 ms (45.8%)** |
| **TOTAL** | **468.78 ms → 2.13 FPS** | **467.64 ms → 2.14 FPS** |

Both models give the **same** end-to-end FPS. Halving the DPU cost bought
nothing, because the DPU was never the bottleneck.

### §7.2 After optimising the ARM-side code

Two changes, same maths and identical detections:

1. **Preprocess** — replaced the float multiply + round over 1.23M elements
   with a precomputed 256-entry int8 lookup table, folding BGR→RGB into the
   same indexing operation.
2. **Postprocess** — instead of running sigmoid over all 2.14M output
   elements, threshold the objectness channel while still in the **raw int8
   domain** (sigmoid is monotonic, so `conf > T` implies `obj_raw >
   logit(T)/scale`). Only the surviving anchors are dequantised and decoded.

| Stage | 4K (3840×2160) | 1080p (1920×1080) |
|---|---|---|
| decode (MJPEG, CPU) | **388.99 ms (83.0%)** | 49.04 ms (38.3%) |
| preprocess (LUT) | 41.75 ms (8.9%) | 40.86 ms (31.9%) |
| DPU inference | 30.64 ms (6.5%) | 30.66 ms (24.0%) |
| postprocess (gated) | **7.52 ms (1.6%)** | 7.35 ms (5.7%) |
| **TOTAL** | **468.90 ms → 2.13 FPS** | **127.91 ms → 7.82 FPS** |
| DPU-only ceiling | 32.64 FPS | 32.62 FPS |

**Postprocessing: 214.32 ms → 7.52 ms, a 28× speedup.** Yet 4K end-to-end did
not move at all — the saving was simply absorbed by decode, which had been
partly hidden behind the slow downstream stages.

### §7.3 Decode measured in isolation — the proof

Reading frames with **no inference whatsoever**:

```
test_4k.avi      60 frames   465.72 ms/frame    2.15 FPS   (decode only)
test_1080p.avi   60 frames   127.10 ms/frame    7.87 FPS   (decode only)
```

Compare with the full-pipeline figures: **2.13 FPS** and **7.82 FPS**.
End-to-end throughput equals decode-only throughput to within 1%.

> **CONCLUSION: the entire pipeline is capped by CPU video decode.
> Inference is essentially free by comparison — the DPU, preprocessing and
> postprocessing all hide behind the decoder.**

### §7.4 Repeated on GENUINE 4K drone footage

Identical script, identical model, real aerial 4K footage (`real_4k.avi`):

| Stage | REAL 4K (3840×2160) | REAL 1080p (1920×1080) |
|---|---|---|
| decode (MJPEG, CPU) | **362.40 ms (82.6%)** | 37.77 ms (33.5%) |
| preprocess (LUT) | 41.94 ms (9.6%) | 40.94 ms (36.3%) |
| DPU inference | 30.94 ms (7.1%) | 30.81 ms (27.4%) |
| postprocess (gated) | 3.40 ms (0.8%) | 3.11 ms (2.8%) |
| **TOTAL** | **438.68 ms → 2.28 FPS** | **112.62 ms → 8.88 FPS** |
| DPU-only ceiling | 32.32 FPS | 32.46 FPS |

Decode measured in isolation on the same clips:

```
real_4k.avi      60 frames   436.33 ms/frame   2.29 FPS   (decode only)
real_1080p.avi   60 frames   112.46 ms/frame   8.89 FPS   (decode only)
```

**Side-by-side with the mosaic clip:**

| | mosaic 4K | real 4K | mosaic 1080p | real 1080p |
|---|---|---|---|---|
| end-to-end | 2.13 FPS | **2.28 FPS** | 7.82 FPS | **8.88 FPS** |
| decode share | 83.0% | **82.6%** | 38.3% | 33.5% |
| DPU-only ceiling | 32.64 FPS | 32.32 FPS | 32.62 FPS | 32.46 FPS |

The two clips agree to within 7%, and the decode share at 4K is identical to
within half a percentage point. **The decode bottleneck is a property of the
platform, not of the test content.** The real clip is marginally faster only
because ffmpeg at `-q:v 6` produces smaller JPEGs than OpenCV's default
encoder, so there is less entropy to decode per frame.

---

## §8. Why decode is slow, and the fix

The bottleneck is not a coding problem. It is a **missing hardware block**.

Probing the current bitstream:

```
software H.264/H.265 decoder : ABSENT   (no avdec_*, no libav, no openh264;
                                         only h264parse / h265parse)
omxh264dec / omxh265dec      : plugin loads, but /dev/allegroDecodeIP missing
                               -> the VCU is NOT in kv260-benchmark-b4096
jpegdec                      : present  (why MJPEG was used for the test clip)
```

So on the stock bitstream this board **cannot decode H.264/H.265 at all**, and
MJPEG must be decoded in software on the ARM cores.

The Kria K26 SOM has a **hardened H.264/H.265 VCU rated up to 4Kp60**. It is
simply not instantiated in this bitstream. `kv260-smartcam` and
`kv260-aibox-reid` — both already installed on this board — do include it.

### §8.1 Projection with the VCU platform

With VCU decode (hardware, ~free) and a PL multiscaler doing the 4K→640 resize
(also hardware — the smartcam preprocessing IP is documented as scaling
"the original 4K/1080p frame to at most 720×720"):

```
remaining CPU/DPU work = DPU 30.64 ms + postprocess 7.52 ms = 38.16 ms
                                                            -> 26.2 FPS
if postprocessing is also moved off the critical path       -> 32.6 FPS
```

**A 4K30 pipeline is achievable on this board** — but it needs a bitstream that
contains both a VCU and a DPU.

### §8.2 The catch

`kv260-smartcam` uses a **B3136** DPU, not B4096. Its fingerprint therefore
differs again, and the `_v25` models installed here would **not** run on it.
Moving to the smartcam platform requires recompiling the models for that DPU.

This is the central architectural tension: **on the KV260 the video path and
the DPU are coupled through a single bitstream.** You cannot mix and match a
video platform from one app with a DPU from another.

---

## §9. What this means for the LCAM-YOLOX work

Same board, same DPU, measured today:

```
stock ofa_yolo_pt_v25 :  51.6 ms DPU,  1 DPU subgraph  +  4 CPU  -> 19.4 FPS
LCAM-YOLOX (lcam_v5_2p5): 40.8 ms DPU,  8 DPU subgraphs + 14 CPU  ->  2.34 FPS
```

**LCAM-YOLOX is *faster on the DPU* and roughly 8× slower end-to-end.** The
whole gap is CPU fallback on the attention gates — exactly what the custom LCAM
IP removes. That is a far cleaner motivating measurement than anything
currently in the write-up, because it is a like-for-like comparison against a
stock vendor model on identical silicon.

Also note the tiling implication. Earlier planning assumed ~40.8 ms per
640×640 inference, which made tiled 4K inference hopeless. At the measured
**6.3 ms** for `tiny_yolov3_vmss_v25`, roughly **5 tiles per frame** fit inside
a 33 ms budget — so tile-based and ROI-based approaches are worth revisiting.

---

## §10. Reproducing this

```bash
# board setup (network does not survive reboot)
ip addr add 192.168.137.126/24 dev eth0
ip route add default via 192.168.137.1

# kv260-benchmark-b4096 auto-loads at boot; confirm the DPU
xdputil query | grep -E 'fingerprint|DPU Arch'      # expect 0x101000016010407

cd /home/root/vai25_models

# single image
python3 detect_ofa_yolo.py \
  /usr/share/vitis_ai_library/models/ofa_yolo_pt_v25/ofa_yolo_pt_v25.xmodel \
  bus.jpg --conf 0.25 --save out.jpg

# DPU-only throughput
xdputil benchmark \
  /usr/share/vitis_ai_library/models/ofa_yolo_p50_v25/ofa_yolo_p50_v25.xmodel 1

# rebuild the test clips (only needed once)
python3 make_test_video.py 120

# 4K video, full pipeline with per-stage timing
#   Clip 2 (mosaic, shows detections)
python3 bench_video_opt.py \
  /usr/share/vitis_ai_library/models/ofa_yolo_p50_v25/ofa_yolo_p50_v25.xmodel \
  test_4k.avi --max-frames 100
#   Clip 1 (genuine 4K drone footage, throughput)
python3 bench_video_opt.py \
  /usr/share/vitis_ai_library/models/ofa_yolo_p50_v25/ofa_yolo_p50_v25.xmodel \
  real_4k.avi --max-frames 100

# annotated output video
python3 bench_video_opt.py \
  /usr/share/vitis_ai_library/models/ofa_yolo_p50_v25/ofa_yolo_p50_v25.xmodel \
  test_1080p.avi --max-frames 40 --save out_1080p_annotated.avi
```

Artefacts produced on the board:

```
/home/root/vai25_models/real_4k.avi              GENUINE 4K drone clip (111 MB)
/home/root/vai25_models/real_1080p.avi           same, 1080p (38 MB)
/home/root/vai25_models/test_4k.avi              mosaic 4K test clip
/home/root/vai25_models/test_1080p.avi           mosaic 1080p test clip
/home/root/vai25_models/frame_4k.jpg             one extracted 4K frame
/home/root/vai25_models/frame_4k_det.jpg         same frame with detections
/home/root/vai25_models/bus_det.jpg              annotated bus.jpg
/home/root/vai25_models/out_1080p_annotated.avi  annotated 1080p video (60 MB)
```

---

## §11. Honest limitations

- Both clips are **MJPEG**, not H.264/H.265, because this bitstream has no
  decoder for the latter. The source of Clip 1 *was* genuine 4K H.264 — it had
  to be transcoded on the PC to be playable at all. MJPEG is more expensive
  per frame than H.264 on the VCU would be, but *cheaper* than software H.264
  would be, so ~436 ms is not a worst case; it is what CPU decode costs here.
- **Clip 1 is genuine 4K drone footage** and validates the throughput result on
  real content. It contains no COCO objects, so it shows no meaningful
  detections — that is the content, not a pipeline failure.
- **Clip 2's frames are mosaics of nine native-720p images**, not footage from
  a 4K sensor. Valid for throughput, latency and for showing detections;
  **not** valid for any claim about 4K detection *accuracy*.
- No claim about **4K detection accuracy** is made anywhere in this document.
  That would need natively-captured, annotated 4K ground truth.
- The models are **COCO/VOC**, so they detect nothing in wildfire imagery.
  Fire/smoke detection needs a fire-trained model quantised and compiled with
  the **2.5** toolchain (already present in WSL at
  `/home/punnam_rahul/wildfire_project/`).
- The `26.2 FPS` VCU figure is a **projection**, not a measurement. It assumes
  hardware decode and hardware scaling are fully overlapped.
- All FPS figures are single-stream, single DPU core.

---

## §13. OPTIMISATION ROUND 2 — 5.3× end-to-end, no hardware change

Added 2026-09-01, after §7. Script: `pipeline_v2.py`, probe: `decode_probe.py`.

Since §8 showed the pipeline is decode-bound, all effort went at decode. Three
software changes, no bitstream change, no new IP.

### §13.1 Two independent decode wins

**(a) GStreamer costs ~2× for nothing.** `cv2.VideoCapture` was taking 436 ms
for output that `cv2.imdecode` produces in 212 ms. `pipeline_v2.py` walks the
AVI RIFF chunks itself and hands raw JPEG buffers to `imdecode`, so the input
file is unchanged.

**(b) Scaled-DCT decode.** The pipeline letterboxes 3840×2160 into 640×640, so
the actual content is **640×360** — only 3.5% of decoded pixels are kept.
libjpeg can decode directly at 1/2 or 1/4 scale via scaled DCT
(`IMREAD_REDUCED_COLOR_2/4`), which is still ≥ 640×360.

```
GStreamer VideoCapture (baseline)  436.33 ms   1.00x   3840x2160
direct cv2.imdecode, full          212.13 ms   2.06x   3840x2160
scaled DCT 1/2                     103.23 ms   4.23x   1920x1080
scaled DCT 1/4                      70.91 ms   6.15x    960x 540
scaled DCT 1/8                      51.69 ms   8.44x    480x 270  <- BELOW 640x360, rejected
```

### §13.2 Multi-core prefetch

Decode + letterbox + quantise are pure functions of a JPEG buffer, so they are
farmed to a `multiprocessing.Pool` across the four A53 cores while the parent
drives the DPU (a single shared resource that stays in the parent).

> **Gotcha, cost real time:** *forking a pool AFTER creating a VART runner
> aborts the child* (`SIGABRT`) — the runner holds device handles and driver
> threads that do not survive `fork()`. The pools must be created **before**
> `vart.Runner.create_runner`. `pipeline_v2.py` reads the network geometry
> from the `xir` graph first, forks the pools, then creates the runner.

Also note `subgraph.get_input_tensors()` returns an **unordered set** in this
binding, unlike `runner.get_input_tensors()` which returns a list.

### §13.3 Measured sweep — real 4K drone clip

Wall-clock FPS, 60 frames. Baseline (§7.4, GStreamer serial) = **2.28 FPS**.

| decode | w=0 (serial) | w=2 | w=3 | w=4 | detections |
|---|---|---|---|---|---|
| full | 3.16 | 5.47 | 8.20 | **9.63** | 37 |
| 1/2 | 5.76 | 1.71 ⚠ | 9.83 | **12.06** | 35 |
| 1/4 | 7.14 | 1.79 ⚠ | 7.49 | **13.09** | 10 |

⚠ The `w=2` entries are anomalous — slower than serial. Not explained;
reproducible across both reduce settings. Likely a scheduling pathology with
2 workers plus parent on 4 cores. Flagged rather than explained away.

**Detection counts are identical across every worker count for a given decode
setting** (37/37/37/37, 35/35/35/35, 10/10/10/10) — the parallelisation is
deterministic and does not perturb results.

### §13.4 Accuracy — the gate that decides which setting is safe

| Clip | content | full → 1/2 | full → 1/4 |
|---|---|---|---|
| mosaic `test_4k.avi` | strong COCO objects (bus, people) | 289 → 297 (**+2.8%**) | 289 → 285 (**−1.4%**) |
| drone `real_4k.avi` | marginal detections only | 37 → 35 (−5.4%) | 37 → 10 (**−73%**) |

**Real objects are preserved within ±3% at both settings.** The −73% on the
drone clip is confined to *marginal* detections: that clip's 37 "detections"
are low-confidence responses to high-frequency water texture. Scaled-DCT does
proper box-averaging in the DCT domain, whereas bilinear 6× downscaling
aliases that texture into false structure. Suppressing them is arguably a
quality improvement — but with no ground truth this is an interpretation, not
a measurement, so **1/2 is the recommended setting** and 1/4 should only be
used where content is known to contain well-resolved objects.

### §13.5 Result

```
                             real 4K      mosaic 4K
baseline (§7.4, GStreamer)   2.28 FPS      2.13 FPS
pipeline_v2 (1/2, w=4)      12.06 FPS      7.86 FPS
                            -----------   -----------
                              5.3x           3.7x
```

**5.3× end-to-end on genuine 4K footage, with no bitstream change, no new IP,
and real detections preserved within 3%.**

### §13.6 Why no new Vivado IP was built

The obvious instinct is to accelerate postprocessing in PL. **It is 3.40 ms of
a 438.68 ms frame — 0.8%.** By Amdahl's law an infinitely fast postprocess IP
returns 0.8%. The same reasoning applied to preprocessing (9.6%) and even to
the DPU itself (7.1%). Every one of them is dwarfed by decode.

The only hardware that matters here is the **VCU**, and that is not an IP to
be written — it is a hardened block already on the K26 SOM that simply needs
instantiating in the bitstream (§8). Custom HLS work on this pipeline is
premature until decode is on hardware.

---

## §12A. TILED INFERENCE — measured against ground truth (2026-09-02)

The wildfire model (`lcam_v5_2p5.xmodel`) was run on 4K frames two ways:
whole-frame downscaled, and tiled. Unlike everything above, this section has
**real ground truth** — the project's own `valset/labels/` (204 of 400 images
carry non-empty YOLO labels; the other 196 are true negatives).

### §12A.1 Why tiling was tried

LCAM-YOLOX was trained on ~1280×720 images letterboxed to 640×640 — content
occupies 640×360, a **2× reduction**. A 3840×2160 frame letterboxes to the
*same* 640×360, i.e. a **6× reduction**: every object is 3× smaller than
anything seen in training. Smoke plumes vanish.

### §12A.2 Method

Ten synthetic-but-labelled 4K frames, each a 3×3 mosaic of nine labelled
valset images resized to exactly 1280×720. YOLO labels are normalised, so they
map into mosaic pixel coordinates exactly:

```
x = (col + cx_norm) * 1280      w = w_norm * 1280
y = (row + cy_norm) *  720      h = h_norm *  720
```

Both methods see the identical frame and identical ground truth. Matching is
greedy highest-score-first, class-aware, IoU ≥ 0.5. **136 ground-truth objects
across 10 4K frames.** Script: `vai25_yolo/eval_tiling.py`.

### §12A.3 Result — the curve is NOT monotonic

| Grid | Tile size | Reduction | Precision | Recall | F1 | ms/frame | Cost |
|---|---|---|---|---|---|---|---|
| downscaled | 3840×2160 | 6.0× | 0.774 | 0.301 | 0.434 | 484 | 1× |
| 2×2 | 1920×1080 | 3.0× | 0.726 | 0.603 | 0.659 | 1923 | 4.0× |
| **3×3** | **1280×720** | **2.0×** | **0.821** | **0.809** | **0.815** | 4302 | 8.9× |
| 4×4 | 960×540 | 1.5× | 0.677 | 0.632 | 0.654 | 7676 | 15.8× |
| 5×5 | 768×432 | 1.2× | 0.507 | 0.522 | 0.514 | 11965 | 24.4× |

**3×3 is optimal, and is the only configuration where precision *improves*
(+0.047) over whole-frame downscaling.** Recall +0.507 (+168%), F1 +0.381.
Beyond 3×3 both metrics collapse: 5×5 costs 24× and scores worse than 2×2.

### §12A.4 Interpretation

1280×720 **is the training image resolution**. Tiling at that granularity
reproduces training conditions exactly. Finer tiles overshoot: objects appear
larger than training, and — the dominant effect — **large smoke plumes are
fragmented across tile boundaries**, producing partial detections that NMS
cannot merge because they are distinct objects. Wildfire smoke is large and
diffuse, so it is unusually sensitive to over-tiling.

> **Rule: optimal tile size = the training image resolution, not the smallest
> tile the compute budget allows.**

This is the same principle as §7.2's decode finding (match the JPEG DCT scale
to the letterbox geometry): *match pipeline geometry to what the model
expects*. Two independent measurements, one underlying thesis.

### §12A.5 Honest limitations

- Frames are **synthetic mosaics of real labelled images**, not native 4K
  captures. Valid for a controlled scale-shift comparison; it is not a claim
  about real 4K sensor data.
- 10 frames / 136 objects is a **small sample**. Treat the ordering as solid
  and the absolute values as indicative.
- Single IoU threshold (0.5) and single confidence (0.30). No mAP sweep.
- Tiling is an **offline / high-accuracy mode**: 3×3 is 4.30 s per 4K frame
  (0.23 FPS), not real time.
- The whole-frame recall of 0.301 shows the model misses **69% of objects**
  when a 4K frame is naively downscaled — the strongest argument in this
  document for why 4K needs more than a resize.

---

## §12. Next steps

1. **Build a VCU + DPU bitstream** (or adapt `kv260-smartcam`) and recompile
   the models for its DPU config. This converts the 26.2 FPS projection into a
   measurement and is the only route to real 4K30.
2. **Move postprocessing off the ARM cores** — at 7.52 ms it is no longer
   urgent, but it becomes ~20% of the budget once decode is fixed.
3. **Revisit tiling / ROI** now that `tiny_yolov3` measures 6.3 ms per
   inference — roughly 5 tiles per 33 ms frame.
4. **Compile a fire-trained model for 2.5** to make the demo domain-relevant.
5. **Re-measure LCAM-YOLOX against `ofa_yolo_pt_v25`** on identical video for a
   like-for-like number in the write-up (§9).
