#!/bin/sh
# =====================================================================
#  4K object detection on KV260 -- stock bitstream, stock models
#  Full demonstration, runs in ~4 minutes.
#
#    sh demo.sh          # normal demo
#    sh demo.sh --full   # adds the 60s xdputil DPU benchmark
# =====================================================================
cd /home/root/vai25_models || exit 1
M=/usr/share/vitis_ai_library/models
P50=$M/ofa_yolo_p50_v25/ofa_yolo_p50_v25.xmodel
PT=$M/ofa_yolo_pt_v25/ofa_yolo_pt_v25.xmodel
Q="grep -v -E GStreamer|WARN"

hdr() {
  echo ""
  echo "======================================================================"
  echo "  $1"
  echo "======================================================================"
}

# ---------------------------------------------------------------- ACT 1
hdr "ACT 1/7  The platform"
echo "--- which bitstream is loaded ---"
xmutil listapps 2>/dev/null | head -2
xmutil listapps 2>/dev/null | awk 'NR>2 && $NF != "-1" {print "  ACTIVE ->", $1}'
echo ""
echo "--- the DPU it contains ---"
xdputil query 2>/dev/null | grep -E '"DPU Arch"|fingerprint|Frequency\(MHz\)|DPU Core Count'
echo ""
echo "--- the runtime driving it ---"
xdputil query 2>/dev/null | grep -E 'libvart-runner'
echo ""
echo ">> NOTE: DPU IP is Vitis-AI 2.5 vintage, runtime is 3.0. Mismatched pair."

# ---------------------------------------------------------------- ACT 2
hdr "ACT 2/7  Why the 250 pre-installed models cannot run"
python3 - <<'PYEOF'
import xir, glob
BOARD = 0x101000016010407
def fp(p):
    g = xir.Graph.deserialize(p)
    s = [x for x in g.get_root_subgraph().toposort_child_subgraph()
         if x.has_attr('dpu_fingerprint')]
    return s[0].get_attr('dpu_fingerprint') if s else None
print("  board DPU fingerprint            : 0x%x" % BOARD)
print("")
print("  PRE-INSTALLED (Vitis-AI 3.0):")
for n in ['yolox_nano_pt','yolov5_nano_pt','yolov6m_pt','ofa_yolo_pt']:
    p = '/usr/share/vitis_ai_library/models/%s/%s.xmodel' % (n, n)
    try:
        f = fp(p)
        print("    %-18s 0x%x   %s" % (n, f, "MATCH" if f == BOARD else "*** REFUSED ***"))
    except Exception:
        pass
print("")
print("  DOWNLOADED (Vitis-AI 2.5):")
for n in ['ofa_yolo_pt_v25','ofa_yolo_p50_v25','tiny_yolov3_vmss_v25']:
    p = '/usr/share/vitis_ai_library/models/%s/%s.xmodel' % (n, n)
    try:
        f = fp(p)
        print("    %-22s 0x%x   %s" % (n, f, "MATCH -> RUNS" if f == BOARD else "REFUSED"))
    except Exception:
        pass
PYEOF

# ---------------------------------------------------------------- ACT 3
hdr "ACT 3/7  Correctness -- canonical YOLOv5 test images"
echo "--- bus.jpg  (expect: 4x person + 1x bus) ---"
python3 detect_ofa_yolo.py $PT bus.jpg --conf 0.25 --save bus_det.jpg 2>/dev/null \
  | sed -n '/DETECTIONS/,$p'
echo ""
echo "--- zidane.jpg  (expect: 2x person + 1x tie) ---"
python3 detect_ofa_yolo.py $PT zidane.jpg --conf 0.25 2>/dev/null \
  | sed -n '/DETECTIONS/,$p'

# ---------------------------------------------------------------- ACT 4
if [ "$1" = "--full" ]; then
hdr "ACT 4/7  DPU-only throughput (60 s)"
echo "--- ofa_yolo_p50_v25, 1 thread ---"
xdputil benchmark $P50 1 2>&1 | grep -E 'FPS=|Test PASS'
else
hdr "ACT 4/7  DPU-only throughput  [SKIPPED - run 'sh demo.sh --full']"
echo "  previously measured: ofa_yolo_pt 19.39 FPS | p50 33.23 FPS | tiny_yolov3 157.81 FPS"
fi

# ---------------------------------------------------------------- ACT 5
hdr "ACT 5/7  4K VIDEO, END-TO-END  (baseline pipeline)"
echo "  real_4k.avi -- genuine 3840x2160 aerial drone footage"
echo ""
python3 bench_video_opt.py $P50 real_4k.avi --max-frames 60 2>/dev/null \
  | sed -n '/frames timed/,$p'

# ---------------------------------------------------------------- ACT 6
hdr "ACT 6/7  PROOF the bottleneck is decode, not the DPU"
echo "--- decode measured ALONE, no inference in the process at all ---"
python3 - <<'PYEOF'
import cv2, time
for v in ['real_4k.avi','real_1080p.avi']:
    cap = cv2.VideoCapture(v)
    for _ in range(3): cap.read()
    n=0; t0=time.time()
    while n < 40:
        ok,_ = cap.read()
        if not ok: break
        n+=1
    dt = time.time()-t0; cap.release()
    print('  %-16s %8.2f ms/frame   %6.2f FPS   (decode only)' % (v, dt/n*1e3, n/dt))
PYEOF
echo ""
echo ">> Compare with ACT 5 end-to-end FPS. They match within ~1%."
echo ">> Everything else runs inside the decoder's shadow and costs nothing."
echo ""
echo "--- and WHY: the hardware video decoder is not in this bitstream ---"
printf "  software H.264 decoder (avdec_h264) : "
gst-inspect-1.0 avdec_h264 >/dev/null 2>&1 && echo "present" || echo "ABSENT"
printf "  VCU hardware decoder device node    : "
[ -e /dev/allegroDecodeIP ] && echo "present" || echo "ABSENT  <-- VCU not instantiated"
printf "  jpegdec (what we fall back on)      : "
gst-inspect-1.0 jpegdec >/dev/null 2>&1 && echo "present" || echo "ABSENT"

# ---------------------------------------------------------------- ACT 7
hdr "ACT 7/7  OPTIMISATION -- 5.3x, no hardware change"
echo "--- decode strategy comparison (pure CPU, no DPU involved) ---"
python3 decode_probe.py real_4k.avi 25 2>/dev/null | sed -n '/DECODE STRATEGIES/,/^$/p'
echo "--- optimised pipeline: bypass GStreamer + scaled-DCT 1/2 + 4 cores ---"
python3 pipeline_v2.py $P50 real_4k.avi --workers 4 --reduce 2 --max-frames 60 2>/dev/null \
  | grep -E 'END-TO-END|detections '

hdr "SUMMARY"
cat <<'EOS'
  4K end-to-end, baseline pipeline      :   2.25 - 2.29 FPS   (very stable)
  4K end-to-end, optimised pipeline     :  12    - 17   FPS   (see note)
  speed-up                              :   5.3x - 7.7x
  DPU-only ceiling (unchanged)          :  32.1  - 32.6 FPS
  DPU share of the baseline frame       :   7.1 %
  decode share of the baseline frame    :  81.5 - 82.6 %

  NOTE: the optimised figure varies run to run (12.06 FPS measured during a
  back-to-back sweep, 17.33 FPS on a freshly booted idle board). It depends
  on how loaded the four A53 cores already are, because decode+preprocess
  are farmed across them. The BASELINE is stable to within 2%, so the
  speed-up is real -- quote the range, not a single number.

  The DPU was never the bottleneck. CPU video decode is.
  The fix is a VCU-enabled bitstream, not faster software or a new IP.
EOS
echo ""
