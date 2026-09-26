# DEMO RUNBOOK — KV260

**This file covers TWO separate demos that use TWO DIFFERENT, mutually
exclusive bitstreams.** The board can only hold one at a time. Check which
track you want *before* you touch the board — loading the wrong one for
your demo undoes the other.

| | Track A — LCAM-YOLOX Hybrid ⭐ | Track B — 4K Video |
|---|---|---|
| **This is** | your actual project: DPU + custom LCAM attention IP | a separate side investigation (2026-09-01), stock silicon, no custom IP |
| Bitstream | `kv260-hybrid2` | `kv260-benchmark-b4096` |
| Fingerprint | `0x101000012010407` | `0x101000016010407` |
| Detects fire/smoke? | **yes** | no (COCO classes only) |
| Custom hardware involved? | **yes** — your accelerator | no |

**If your professor is here to see your project, use Track A.** Track B is
kept below only because it was already fully written and verified — do not
run it by accident (its own Act 0.3 will *unload* Track A's bitstream).

---

# TRACK A — LCAM-YOLOX Hybrid Demo ⭐ (use this one)

**Verified working. Every number below was measured on hardware and is
recorded in `PROJECT_HISTORY.md` §31–§40.** This is DPU + your custom
LCAM attention accelerator, in ONE bitstream, loaded ONCE — no platform
switching happens anywhere in this track.

## A0. Pre-flight (5 minutes before the demo)

### A0.1 Power on and restore networking

Does **not** survive reboot. On the board's console (monitor+keyboard or
serial):

```sh
ip addr add 192.168.137.126/24 dev eth0
ip route add default via 192.168.137.1
```

If those answer `RTNETLINK answers: File exists`, the board never actually
rebooted — the address is already there, which is fine.

### A0.2 From the PC, confirm it answers

```powershell
& "C:\Program Files\PuTTY\plink.exe" -ssh -batch `
    -hostkey "SHA256:YNosFSx1Q/13iPt+28J9WWUiLAvYBvv3UhA6i7vifa0" `
    -pw "<BOARD_PASSWORD>" root@192.168.137.126 "uptime"
```

If the PC's Ethernet adapter shows *Disconnected*, the board is off or
unplugged — check that first, not the software.

### A0.3 CRITICAL — load the hybrid bitstream and confirm it

```sh
xmutil unloadapp
xmutil loadapp kv260-hybrid2
sleep 3
xdputil query | grep fingerprint
```

| You see | Meaning | Action |
|---|---|---|
| `0x101000012010407` | ✅ hybrid loaded, proceed | — |
| `0x101000016010407` | Track B's stock bitstream is loaded | you just ran the fix above — re-check |
| anything else / error | nothing loaded | re-run `xmutil loadapp kv260-hybrid2` |

### A0.4 Set the two environment variables every command below needs

```sh
export XLNX_VART_FIRMWARE=/lib/firmware/xilinx/kv260-hybrid2/hybrid.xclbin
export XRT_INI_PATH=/home/root/xrt.ini
cd ~/hybrid_pkg
```

**`XRT_INI_PATH` is not optional.** Without it every DPU call times out
after 10 s (the platform cascades all compute-unit interrupts onto one
line XRT never programs — polling sidesteps it entirely, costs nothing
measurable). Confirm the file is right if in doubt:

```sh
cat /home/root/xrt.ini
#   [Runtime]
#   ert_polling=true
```

### A0.5 Known transient — do not panic if it happens

**The first DPU call after a fresh `loadapp` sometimes times out.** If any
command below prints `cu timeout!`:

```sh
xmutil unloadapp
xmutil loadapp kv260-hybrid2
sleep 3
```

then re-run the same command. It works on the second try. This is
documented and expected (§33.2) — not a sign anything is broken.

### ⚠ A0.6 One command you must NEVER run live

**Do not run `measure_power.py`.** It once left the board with an
unkillable process that only a hard power cycle could clear (§39.8). Power
numbers are already measured — quote them (Act 5) instead of re-running.

---

## The demo — 5 acts, ~5 minutes

### ACT 1 — One platform, two compute units ⭐ answers "did you switch platforms?"

```sh
xbutil examine -r all 2>/dev/null | grep -B2 -A8 "Compute Units"
```

**Expect** — both listed together, from the SAME xclbin, at the SAME time:
```
Index   Name                                              Base_Address    Status
0       lcam_attention_gate_opt:lcam_attention_gate_opt_1  0xa0020000      (IDLE)
1       DPUCZDX8G:DPUCZDX8G_1                              0xa0010000      (IDLE)
```

**Say:** one bitstream, one `xclbin`, two compute units side by side on the
same fabric — the vendor DPU and a custom accelerator I designed, wired
into the same platform. No switching between them at any point.

---

### ACT 2 — Correctness: the accelerator is bit-for-bit identical to software

```sh
python3 hybrid_pipeline.py --gates ip    --round ref --runs 1 --save /tmp/a.npy
python3 hybrid_pipeline.py --gates numpy --round ref --runs 1 --save /tmp/b.npy
python3 ab_compare.py /tmp/a.npy /tmp/b.npy
```

**Expect**
```
max|diff|        = 0
mean|diff|       = 0
mismatched elems = 0 of 58800
-> IDENTICAL - the IP reproduces the reference exactly
```

**Say:** same pipeline, run twice — once with the attention gates on my
custom silicon, once with the identical arithmetic in software. Every
other line of code is shared, so this isolates the accelerator exactly:
**zero elements differ.**

---

### ACT 3 — Speed: single-frame latency and where the time goes

```sh
python3 hybrid_pipeline.py --gates ip --round ref --runs 5
```

**Expect**
```
DPU  (8 subgraphs)     :   ~43 ms
gates (4)              :   ~26 ms
other CPU subgraphs    :    ~3 ms
mean total             :   ~76 ms   -> ~13.2 FPS
GraphRunner baseline   :  426.51 ms   -> 2.34 FPS
speedup                :    ~5.6x
```

**Say:** the stock Vitis-AI runtime falls back to the ARM CPU for the
attention gates because the DPU cannot execute that operator — 385 ms of
every 427 ms frame. My accelerator does the same four gates in 1.8 ms.
**5.6× faster end to end, same board, same model, one line of code
different.**

---

### ACT 4 — Image in, boxes out — ONE command

```sh
cd ~/hybrid_pkg
python3 hybrid_pipeline.py --gates ip --runs 5 --out ~/demo_result.jpg
```

**Expect** — the same FPS block as Act 3, followed by:
```
  postprocessing (decode + NMS + draw)...
  concat order: 80,40,20   anchors above conf=0.30: 15
  fire    conf=0.875  box=[378, 5, 769, 477]
  smoke   conf=0.651  box=[469, 418, 713, 494]
  smoke   conf=0.301  box=[607, 513, 634, 541]
  TOTAL DETECTIONS: 3
  annotated image written to /home/root/demo_result.jpg
  postprocess cost: decode 15.5 + nms/draw 26.3 + imwrite 25.7
  = ~68 ms  (NOT counted in the hardware FPS above)
```

**Say:** `--out` runs decode + NMS + drawing right after the timed
inference loop, in the same process — no `.npz` round-trip, no second
command. **The FPS printed above is unaffected**: postprocessing is timed
separately and printed on its own line so the cost is visible, not
hidden. It is real (~68 ms, mostly JPEG encoding on the ARM core) but it
happens once, after the 5 timed inference runs are already done — it does
not enter the FPS average.

Serve the annotated image so your professor can see it on their own
screen:
```sh
cd ~ && python3 -m http.server 8080
```
then open `http://192.168.137.126:8080/demo_result.jpg` in a browser.

For **your own image**:
```sh
python3 hybrid_pipeline.py --gates ip --runs 5 --image ~/myphoto.jpg --out ~/demo_result.jpg
```

(The old two-command form — `--save-npz` then a separate
`postprocess_detections.py --npz ...` call — still works and is useful if
you want to re-run decoding/NMS with different `--conf`/`--nms` values
without re-running inference on the board.)

---

### ACT 4b — Video in, boxes out — real fire/smoke footage, ONE command

A real video (built from the same 400 labelled photos used for the mAP
result — see `PROJECT_HISTORY.md` §42, no genuine fire/smoke video existed
before this) is already staged on the board:

```sh
cd ~/hybrid_pkg
python3 video_throughput_hybrid.py /home/root/wildfire_real_720p.avi \
        --save ~/wildfire_demo.avi
```

Runtime: **~45 seconds for all 400 frames**, with a progress line every 50
frames so it's visible your professor isn't watching a hung terminal.
**Expect**
```
fingerprint check OK: 0x101000012010407 (hybrid bitstream loaded)
engine ready: 8 DPU subgraphs, 10 LUT/int8-fast CPU subgraphs, gates=ip
video: /home/root/wildfire_real_720p.avi   400 frames to process
  50 / 400 frames  (...)
  ...
  400 / 400 frames  (...)

  decode                     :    1.8 ms/frame
  preprocess                 :   25.8 ms/frame
  DPU (8 subgraphs)          :   43.0 ms/frame
  LCAM gate (4)              :   27.4 ms/frame
  other CPU subgraphs        :    2.7 ms/frame
  postprocess (decode+NMS)   :    7.3 ms/frame
  draw+encode (--save only)  :    2.1 ms/frame
END-TO-END, camera to boxes   :  115.8 ms/frame ->  8.6 FPS
total detections in this clip : 454   (196/400 frames with >=1)
annotated video written to: /home/root/wildfire_demo.avi
```

**Say:** this is the FIRST genuinely camera-to-boxes number in the whole
project — every other FPS figure (13.2, 18.7, 19.4) reused one preloaded
image and never paid real video decode/preprocess/postprocess. On a real
video this ships at **~8.9 FPS**, not because the accelerator got slower,
but because ~34 ms/frame of real, previously-unmeasured decode+preprocess
+postprocess is now included. Say which number you're quoting.

**If short on time**, run a shorter clip instead and quote the full-clip
figures above from memory:
```sh
python3 video_throughput_hybrid.py /home/root/wildfire_real_720p.avi \
        --max-frames 60 --save ~/wildfire_demo_short.avi
```
(~7 seconds, same per-frame numbers, fewer detections since it only covers
the first 60 of 400 source images).

**Pull and view it** exactly like the image demo:
```powershell
& "C:\Program Files\PuTTY\pscp.exe" -scp -batch -pw "<BOARD_PASSWORD>" `
    -hostkey "SHA256:YNosFSx1Q/13iPt+28J9WWUiLAvYBvv3UhA6i7vifa0" `
    root@192.168.137.126:/home/root/wildfire_demo.avi .\wildfire_demo.avi
start .\wildfire_demo.avi
```
(If the destination folder doesn't exist yet, `pscp` fails with
`Cannot create file` — run from a real folder, or `cd` there first.)

A pre-built copy with real detections already drawn is also sitting in
this package at `demo_output\wildfire_real_720p_det.avi` if you want to
open it without touching the board at all.

**If asked "why does the accelerator only give 3× on video instead of
5.6×?"** — the compute-only figures (75.8 ms / 426.5 ms) never included
video decode/preprocess/postprocess; on real video the SOFTWARE-gates path
pays that same ~34 ms too, so the comparison is 112.0 ms vs 326.6 ms
end-to-end = **2.92×**, still large, just answering a different question
than the compute-only number. Both are correct — say which one you mean.

**If asked "why did it label that smoke plume as fire?"** — measured, not
guessed (§42.7bis of `PROJECT_HISTORY.md`): only 0.6% of detections that
land on a real object get the wrong class. The two confirmed instances in
this dataset are both images of blazing, fully-involved fire where the
GROUND TRUTH itself is labelled "smoke" — a labelling inconsistency in the
training data, not a proven model weakness. Reproduce with:
```sh
python3 eval_map.py --gates ip --conf 0.30 --confusion
```

---

### ACT 5 — Accuracy, throughput, power *(quote or run live — your call on time)*

**Accuracy on 400 labelled images** (~55 s to run live):
```sh
python3 eval_map.py --gates ip
```
**Expect**
```
class    objects   AP@0.5   AP@.5:.95
fire         208   0.7378      0.4103
smoke        292   0.7342      0.3446
mAP                0.7360      0.3774
```
**Say:** measured on 400 labelled validation images, not one lucky demo
photo. Running the same 400 images with the gates in software gives the
**identical** mAP — the accelerator changes speed, not accuracy.

**Throughput with multiple frames in flight** (~10 s to run live):
```sh
python3 pipelined_throughput.py --workers 2 --frames 40
```
**Expect** `~55 ms/frame -> ~18.7 FPS` (latency per frame rises to
~107 ms — throughput and latency trade off; say which one you're quoting).

**For the 19-FPS ceiling (4 workers)** — the number that saturates:
```sh
python3 pipelined_throughput.py --workers 4 --frames 40
```
**Expect** `~52 ms/frame -> ~19.3 FPS` (per-frame latency now ~206 ms —
worse latency for marginal throughput gain over 2 workers; explain this
trade-off if asked, don't just quote the bigger number).

Every worker processes the SAME preloaded frame — this is a throughput
microbenchmark, not a live multi-image demo, so it intentionally has no
per-frame decode/draw cost inside the timed loop (consistent with how
every FPS number in this project is defined: hardware compute only).

**If you also want to SEE the detections from a pipelined run**, add
`--out` — it decodes and draws one representative result (all workers see
the identical frame, so any one result is representative) right after the
timing loop finishes, exactly like Act 4:
```sh
python3 pipelined_throughput.py --workers 4 --frames 40 --out ~/demo_4w.jpg
```
This adds nothing to the reported FPS — same reasoning as Act 4.

**Power — quote, do NOT re-run measure_power.py:**
```
measured board power, idle    : 4.81 W
measured board power, running : 6.36 W
Vivado's own PL-only estimate  : 6.90 W  <- exceeds the WHOLE board's
                                            measured draw, so it is
                                            provably too high; the
                                            measured 6.36 W is the
                                            honest number
energy per frame, hybrid      : 0.44 J
energy per frame, stock       : 2.28 J   -> 5.2x more efficient per frame
```

---

## Closing summary

```
end-to-end, stock DPU + CPU fallback   : 426.51 ms    2.34 FPS
end-to-end, hybrid (this project)      :  ~76 ms      ~13.2 FPS    5.6x
throughput, pipelined (2 workers)      :  ~55 ms      ~18.7 FPS
custom LCAM gate, on silicon           :   1.82 ms    bit-exact (0/58800 diff)
detection accuracy (mAP@0.5)           :   0.7360     unchanged by the IP
board power under load (measured)      :   6.36 W
energy per inference                   :   0.44 J     5.2x better than stock

One bitstream. One xclbin. Two compute units, DPU and custom silicon,
side by side. No platform switch anywhere in this demo.
```

---

## Questions your professor will probably ask (Track A)

**"Did you actually change platforms between the fast and slow numbers?"**
No — see Act 1. Both the 426.51 ms baseline and the 76 ms hybrid result run
on the *same loaded bitstream*; only which code path handles the attention
gates differs (`--gates numpy` vs `--gates ip` in the scripts above).

**"Why not put all nine of your custom IPs on the chip?"**
Tried it (§40). At 7 compute units the design still had 23% of its LUTs
free but 99.96% of its slices full, and timing failed by 1.5 ns. At 10 it
didn't fit at all — wiring 23 AXI masters costs more logic than the
kernels themselves. **On this device there is room for the DPU plus
roughly one custom accelerator** — which is exactly the one operator the
DPU cannot run natively. That's not a shortfall; it's the answer to "how
many can fit," measured.

**"Is the speed-up real or did you change the algorithm?"**
Act 2 is the answer: same code, same weights, only the gate execution
target differs, output is bit-identical to 0 difference across 58,800
elements.

**"Why is the accelerator only 1.8 ms but the frame only got 5.6× faster,
not 100×?"**
Because the gates were never the whole frame — the DPU's own convolution
work is ~43 ms and doesn't change. The accelerator removed the *CPU
fallback* (385 ms), not the DPU's real work.

**"What's left to improve?"**
Reading the accelerator's result back to the CPU costs ~22 ms per frame
because of how the platform maps that memory (write-combining, not
cached). Three fixes were tried and each is documented as a negative
result with evidence (§37) — it is a platform property, not something we
overlooked.

---

## If something goes wrong (Track A)

| Symptom | Cause | Fix |
|---|---|---|
| `cu timeout!` on any DPU call | known transient after a fresh `loadapp` (§33.2) | `xmutil unloadapp; xmutil loadapp kv260-hybrid2`, wait 3 s, retry |
| fingerprint shows `...16010407` | Track B's stock bitstream is loaded | `xmutil unloadapp; xmutil loadapp kv260-hybrid2` |
| `Connection timed out` / adapter *Disconnected* | board off or cable out | power on, re-apply §A0.1 networking |
| board becomes unresponsive to `xmutil` (any command hangs) | a process is stuck in kernel D-state (§39.8) — this is NOT recoverable by software | **hard power cycle** — pull power, restore, redo §A0.1 |
| `ab_compare.py` reports a mismatch | should never happen — stop and report it, don't re-run hoping it clears | re-check `--round ref` is passed to both runs |

---

# TRACK B — 4K Video Demo (stock silicon, no custom IP)

⚠ **This track loads a DIFFERENT bitstream (`kv260-benchmark-b4096`) and
will unload Track A's hybrid design.** Only run this section if you
specifically want to show the 4K-video / CPU-decode investigation instead
of your LCAM-YOLOX project. Do not run both tracks back-to-back without
re-loading between them.

**Everything here was executed successfully on the board on 2026-09-01.**
Expected outputs are the real values measured that day. If you see numbers
close to these, it is working.

**`demo.sh` has now been tested end to end on the board — all seven acts run
clean.** Both paths work: run `demo.sh` for a hands-off demo, or the individual
commands below if you want to narrate each step.

---

## B0. Pre-flight (5 minutes before the demo)

### B0.1 Power on and restore networking

Networking does **not** survive reboot. On the board's console
(monitor+keyboard, or serial):

```sh
ip addr add 192.168.137.126/24 dev eth0
ip route add default via 192.168.137.1
```

### B0.2 From the PC, confirm it answers

```powershell
& "C:\Program Files\PuTTY\plink.exe" -ssh -batch `
    -hostkey "SHA256:YNosFSx1Q/13iPt+28J9WWUiLAvYBvv3UhA6i7vifa0" `
    -pw "<BOARD_PASSWORD>" root@192.168.137.126 "uptime"
```

If the PC's Ethernet shows *Disconnected*, the board is off or unplugged —
check that first, not the software.

### B0.3 CRITICAL — confirm the right bitstream is loaded

**This is the single most likely thing to break the demo.** The board can only
hold one bitstream, and if `kv260-hybrid2` is loaded (Track A) none of these
models will run.

```sh
xdputil query | grep fingerprint
```

| You see | Meaning | Action |
|---|---|---|
| `0x101000016010407` | `kv260-benchmark-b4096` | ✅ correct, proceed |
| `0x101000012010407` | `kv260-hybrid2/3` (Track A) | ❌ run the fix below |

Fix (safe, reversible, **never** use JTAG):

```sh
xmutil unloadapp
xmutil loadapp kv260-benchmark-b4096
xdputil query | grep fingerprint          # re-check
```

### B0.4 Confirm the assets are present

```sh
cd /home/root/vai25_models && ls
```

Expect: `real_4k.avi`, `real_1080p.avi`, `test_4k.avi`, `test_1080p.avi`,
`bus.jpg`, `zidane.jpg`, and the scripts
`detect_ofa_yolo.py bench_video_opt.py decode_probe.py pipeline_v2.py`.

```sh
ls -d /usr/share/vitis_ai_library/models/*_v25
```
Expect three: `ofa_yolo_pt_v25`, `ofa_yolo_p50_v25`, `tiny_yolov3_vmss_v25`.

---

## The demo — seven acts, ~4 minutes

Set these once so the commands stay short:

```sh
cd /home/root/vai25_models
M=/usr/share/vitis_ai_library/models
PT=$M/ofa_yolo_pt_v25/ofa_yolo_pt_v25.xmodel
P50=$M/ofa_yolo_p50_v25/ofa_yolo_p50_v25.xmodel
```

---

### ACT 1 — The platform

```sh
xmutil listapps | head -2
xdputil query | grep -E '"DPU Arch"|fingerprint|Frequency|Core Count'
xdputil query | grep libvart-runner
```

**Expect**
```
"DPU Arch":"DPUCZDX8G_ISA1_B4096_0101000016010407"
"DPU Frequency (MHz)":300
"DPU Core Count":1
"fingerprint":"0x101000016010407"
"libvart-runner.so":"... Version: 3.0.0 ..."
```

**Say:** one DPU core, B4096 at 300 MHz. The DPU IP is Vitis-AI **2.5**
vintage (v4.0.0, built 2022-05-14) but the runtime is **3.0** — a mismatched
board image.

---

### ACT 2 — Why 250 pre-installed models cannot run

```sh
python3 - <<'EOF'
import xir
B = 0x101000016010407
def fp(n):
    p='/usr/share/vitis_ai_library/models/%s/%s.xmodel'%(n,n)
    g=xir.Graph.deserialize(p)
    s=[x for x in g.get_root_subgraph().toposort_child_subgraph()
       if x.has_attr('dpu_fingerprint')]
    return s[0].get_attr('dpu_fingerprint')
print("board DPU: 0x%x\n"%B)
for n in ['yolox_nano_pt','yolov5_nano_pt','yolov6m_pt']:
    print("  3.0  %-22s 0x%x  %s"%(n,fp(n),"MATCH" if fp(n)==B else "*** REFUSED ***"))
for n in ['ofa_yolo_pt_v25','ofa_yolo_p50_v25']:
    print("  2.5  %-22s 0x%x  %s"%(n,fp(n),"MATCH -> RUNS" if fp(n)==B else "REFUSED"))
EOF
```

**Expect** the three 3.0 models at `0x101000056010407` → **REFUSED**, and the
two `_v25` models at `0x101000016010407` → **MATCH → RUNS**.

**Say:** the fingerprint is a hash of the DPU's *feature configuration*, not a
version number. All of them report the same target *name*
`DPUCZDX8G_ISA1_B4096` — only the fingerprint separates them. AMD changed the
default B4096 config between 2.5 and 3.0.

⚠ **If asked "why not just disable the check?"** — `XLNX_ENABLE_FINGERPRINT_CHECK=0`
is all over the internet. It runs and gives silent wrong answers. Do not use it.

---

### ACT 3 — Correctness

```sh
python3 detect_ofa_yolo.py $PT bus.jpg --conf 0.25 --save bus_det.jpg
python3 detect_ofa_yolo.py $PT zidane.jpg --conf 0.25
```

**Expect** — these are the canonical YOLOv5 results:
```
bus.jpg     person 0.909, person 0.893, person 0.856, bus 0.746, person 0.495
zidane.jpg  person 0.900, tie 0.787, person 0.787
```

`bus_det.jpg` is written with boxes drawn — open it to show the geometry is right.

---

### ACT 4 — DPU-only throughput  *(60 s per model — consider skipping live)*

```sh
xdputil benchmark $P50 1
```

**Expect** `FPS= 33.2` and `Test PASS.`

Previously measured, quote rather than re-run:

| model | input | FPS | ms |
|---|---|---|---|
| `ofa_yolo_pt_v25` | 640² | 19.39 | 51.6 |
| `ofa_yolo_p50_v25` | 640² | 33.23 | 30.1 |
| `tiny_yolov3_vmss_v25` | 416² | 157.81 | 6.3 |

**Say:** the DPU alone clears 30 FPS at 640×640. It is not the problem.

---

### ACT 5 — 4K video, end to end  ⭐ the main result

```sh
python3 bench_video_opt.py $P50 real_4k.avi --max-frames 60
```

**Expect** (~30 s to run)
```
video : real_4k.avi  3840x2160
  decode (MJPEG,CPU):   362.40    82.6%
  preprocess  (LUT) :    41.94     9.6%
  DPU inference     :    30.94     7.1%
  postprocess (gated):    3.40     0.8%
  TOTAL             :   438.68
END-TO-END          : 2.28 FPS
DPU-only ceiling    : 32.32 FPS
```

**Say:** genuine 4K aerial drone footage, video file in → detections out, no
manual steps. 2.28 FPS — but look at the split. **The DPU is 7%. Decode is 83%.**

---

### ACT 6 — Prove decode is the bottleneck

```sh
python3 - <<'EOF'
import cv2, time
for v in ['real_4k.avi','real_1080p.avi']:
    cap=cv2.VideoCapture(v)
    for _ in range(3): cap.read()
    n=0; t0=time.time()
    while n<40:
        ok,_=cap.read()
        if not ok: break
        n+=1
    dt=time.time()-t0; cap.release()
    print('%-16s %8.2f ms/frame  %6.2f FPS (decode only)'%(v,dt/n*1e3,n/dt))
EOF
```

**Expect**
```
real_4k.avi        436.33 ms/frame    2.29 FPS (decode only)
real_1080p.avi     112.46 ms/frame    8.89 FPS (decode only)
```

**Say:** decode alone, with no inference in the process at all, gives 2.29 FPS.
The full pipeline gives 2.28. **Within 0.4%.** Everything else — preprocessing,
the DPU, postprocessing — runs inside the decoder's shadow and costs nothing.

Then show *why*:

```sh
gst-inspect-1.0 avdec_h264 >/dev/null 2>&1 && echo "H264 sw decoder: present" || echo "H264 sw decoder: ABSENT"
ls /dev/allegroDecodeIP 2>/dev/null || echo "VCU device node: ABSENT -- not in this bitstream"
gst-inspect-1.0 jpegdec >/dev/null 2>&1 && echo "jpegdec: present"
```

**Say:** there is **no** H.264/H.265 decoder of any kind — this bitstream
cannot decode H.264 at all. The K26 SOM has a hardened **4Kp60 VCU**; it is
simply not instantiated here. That is the missing piece.

---

### ACT 7 — Optimisation: 5.3×, no hardware change

```sh
python3 decode_probe.py real_4k.avi 25
```

**Expect**
```
full decode (IMREAD_COLOR)           212.13 ms/frame    4.71 FPS -> (2160,3840,3)
scaled DCT 1/2 (REDUCED_COLOR_2)     103.23 ms/frame    9.69 FPS -> (1080,1920,3)
scaled DCT 1/4 (REDUCED_COLOR_4)      70.91 ms/frame   14.10 FPS -> (540,960,3)
scaled DCT 1/8 (REDUCED_COLOR_8)      51.69 ms/frame   19.35 FPS -> (270,480,3)
```

**Say:** two separate wins. GStreamer's `VideoCapture` costs 436 ms for output
`cv2.imdecode` produces in 212 ms — 2× overhead for nothing. And we letterbox
4K into 640×640, so the real content is only **640×360** — we keep 3.5% of the
pixels. libjpeg decodes directly at ½ scale via scaled DCT, still well above
640×360, so nothing the pipeline would have kept is lost. (⅛ gives 480×270,
*below* 640×360, so it would upscale — rejected.)

```sh
python3 pipeline_v2.py $P50 real_4k.avi --workers 4 --reduce 2 --max-frames 60
```

**Expect** `wall-clock` between **12 and 17 FPS**, detections `35`.

⚠ **This number varies run to run** — 12.06 FPS measured during a back-to-back
sweep, 17.33 FPS on a freshly booted idle board. Decode+preprocess are farmed
across the four A53 cores, so it depends on how loaded they already are. The
*baseline* (Act 5) is stable to within 2%, so the speed-up is real. **Quote the
range, 5.3×–7.7×, or the reproducible 12.06 FPS figure — not the one-off 17.33**
— if you claim 17 and it prints 12, that looks like you don't know your own
result.

**Say:** bypass GStreamer + scaled-DCT ½ + decode/preprocess farmed across all
four A53 cores while the DPU runs. **2.28 → 12.06 FPS, 5.3×, no hardware
change.**

---

## Closing summary (Track B)

```
  4K end-to-end, baseline               :   2.25 - 2.29 FPS   (stable to 2%)
  4K end-to-end, optimised              :  12    - 17   FPS
  speed-up                              :   5.3x - 7.7x
  DPU-only ceiling (never the limit)    :  32.1  - 32.6 FPS
  DPU share of the baseline frame       :   7.1 %
  decode share of the baseline frame    :  81.5 - 82.6 %

  The DPU was never the bottleneck. CPU video decode is.
  The fix is a VCU-enabled bitstream -- not faster software, not a new IP.
```

---

## Questions your professor will probably ask (Track B)

**"Is that a real 4K video or did you fake it?"**
Real. 3840×2160 H.264 aerial drone footage, 677 frames, 28.2 s. It had to be
transcoded to MJPEG on the PC because the board cannot decode H.264 at all.
There is also a second clip (`test_4k.avi`) built as an exact 3×3 mosaic of
nine *native* 1280×720 images — no upscaling. Both give the same answer within
7%, which is the point: the bottleneck is the platform, not the content.

**"Why not just accelerate it with a custom IP in the FPGA?"**
Postprocessing is 3.40 ms of a 438.68 ms frame — 0.8%. An infinitely fast
postprocess IP returns 0.8%. Preprocess is 9.6%, the DPU 7.1%. Every candidate
is dwarfed by decode. The only hardware that matters is the VCU, and that is
not an IP you write — it is a hardened block that needs instantiating.

**"So just load kv260-smartcam, it has the VCU."**
It does — but it uses a **B3136** DPU, not B4096. Different fingerprint, so
these models would not run on it. The video path and the DPU share one
bitstream; moving to 4K forces a matched rebuild of both plus a recompile of
every model. That is the real architectural cost.

**"Does the ½-scale decode hurt accuracy?"**
On content with strong objects, no: 289 → 297 detections (+2.8%) on the mosaic
clip. On the drone clip it drops 37 → 35. The one alarming number is ¼-scale
on the drone clip (37 → 10), but those were low-confidence responses to water
texture — scaled DCT box-averages properly while bilinear 6× downscaling
aliases that texture into false structure. **½ is the recommended setting.**
No ground-truth mAP was measured, so this is characterised, not proven.

**"Why is your model detecting nothing in the wildfire images?"**
Because these are COCO models — 80 classes, none of which is fire or smoke.
That is a correct negative, not a failure. Fire detection needs a fire-trained
model quantised and compiled with the **2.5** toolchain — which is exactly
what Track A above is.

---

## If something goes wrong (Track B)

| Symptom | Cause | Fix |
|---|---|---|
| `CHECK fingerprint fail! ... 0x101000012010407` | `kv260-hybrid2` (Track A) is loaded | `xmutil unloadapp; xmutil loadapp kv260-benchmark-b4096` |
| `Connection timed out` / adapter *Disconnected* | board off or cable out | power on, re-apply §B0.1 networking |
| `cannot open video` | wrong directory | `cd /home/root/vai25_models` |
| Python `Aborted` (SIGABRT) in `pipeline_v2.py` | pool forked after VART runner | already handled in the script; do not reorder it |
| Stray `GStreamer-CRITICAL` lines | harmless OpenCV/GStreamer noise | append `2>/dev/null` |
| `xdputil benchmark` seems hung | it runs a fixed 60 s | wait, or skip Act 4 |

---

## Optional: the one-command wrapper (Track B only)

`demo.sh` runs all seven Track B acts unattended in ~4 minutes. **Tested
working on the board.** It is already at `/home/root/vai25_models/demo.sh`.
If you ever need to re-copy it:

```powershell
& "C:\Program Files\PuTTY\pscp.exe" -scp -batch `
    -hostkey "SHA256:YNosFSx1Q/13iPt+28J9WWUiLAvYBvv3UhA6i7vifa0" -pw "<BOARD_PASSWORD>" `
    "<path>\vai25_yolo\demo.sh" root@192.168.137.126:/home/root/vai25_models/
```

```sh
cd /home/root/vai25_models
sh demo.sh            # ~4 min
sh demo.sh --full     # adds the 60 s DPU benchmark
```

To suppress harmless OpenCV/GStreamer warning lines:

```sh
sh demo.sh 2>&1 | grep -v -E 'GStreamer-CRITICAL|WARN:0|gst_query_set|assertion'
```

**Run it once yourself before the demo** so you know what scrolls past.

---

## Reference

- **Track A (LCAM-YOLOX hybrid):** full findings, methodology, and every
  negative result with evidence in `PROJECT_HISTORY.md` §31–§40. Scripts on
  the board at `~/hybrid_pkg/`.
- **Track B (4K video):** `4K_VIDEO_RESEARCH.md` — full findings, §1–§13,
  with methodology and limitations. Scripts in `vai25_yolo/`. Board:
  `/home/root/vai25_models/`.
