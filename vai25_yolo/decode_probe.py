#!/usr/bin/env python3
"""
Decode-strategy probe.

The pipeline downscales 3840x2160 -> 640x640, i.e. it discards ~97% of the
decoded pixels. libjpeg can decode directly at 1/2, 1/4 or 1/8 scale using
scaled-DCT, at a fraction of the cost. OpenCV exposes this through
IMREAD_REDUCED_COLOR_{2,4,8}.

This script pulls the raw JPEG frames straight out of the MJPEG AVI (so the
input file is unchanged) and times each decode strategy.

  python3 decode_probe.py <video.avi> [n_frames]
"""
import sys, time, struct
import numpy as np
import cv2


def avi_jpeg_frames(path, limit=None):
    """Yield raw JPEG byte strings from an MJPEG .avi, via RIFF chunk walk."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"RIFF" or data[8:12] != b"AVI ":
        raise RuntimeError("not a RIFF/AVI file")
    # locate the 'movi' LIST payload
    i = data.find(b"movi")
    if i < 0:
        raise RuntimeError("no movi list found")
    p = i + 4
    n = len(data)
    out = []
    while p + 8 <= n:
        cid = data[p:p + 4]
        try:
            sz = struct.unpack("<I", data[p + 4:p + 8])[0]
        except struct.error:
            break
        p += 8
        if sz < 0 or p + sz > n:
            break
        if cid[2:4] in (b"dc", b"db"):
            chunk = data[p:p + sz]
            if chunk[:2] == b"\xff\xd8":
                out.append(chunk)
                if limit and len(out) >= limit:
                    return out
        p += sz + (sz & 1)          # chunks are word-aligned
    return out


def bench(name, fn, bufs, reps=1):
    # warm-up
    for b in bufs[:3]:
        fn(b)
    t0 = time.time()
    shp = None
    for _ in range(reps):
        for b in bufs:
            img = fn(b)
            if img is not None:
                shp = img.shape
    dt = (time.time() - t0) / (reps * len(bufs))
    print("  %-34s %8.2f ms/frame  %7.2f FPS   -> %s"
          % (name, dt * 1e3, 1.0 / dt, shp))
    return dt, shp


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "real_4k.avi"
    nf = int(sys.argv[2]) if len(sys.argv) > 2 else 40

    print("extracting raw JPEG frames from %s ..." % path)
    bufs = avi_jpeg_frames(path, nf)
    if not bufs:
        print("no JPEG frames recovered"); return 1
    sizes = [len(b) for b in bufs]
    print("  %d frames, mean %.2f MB per JPEG\n" % (len(bufs), np.mean(sizes) / 1e6))

    arrs = [np.frombuffer(b, dtype=np.uint8) for b in bufs]

    print("DECODE STRATEGIES")
    r = {}
    r["full"] = bench("full decode (IMREAD_COLOR)",
                      lambda a: cv2.imdecode(a, cv2.IMREAD_COLOR), arrs)
    r["r2"] = bench("scaled DCT 1/2 (REDUCED_COLOR_2)",
                    lambda a: cv2.imdecode(a, cv2.IMREAD_REDUCED_COLOR_2), arrs)
    r["r4"] = bench("scaled DCT 1/4 (REDUCED_COLOR_4)",
                    lambda a: cv2.imdecode(a, cv2.IMREAD_REDUCED_COLOR_4), arrs)
    r["r8"] = bench("scaled DCT 1/8 (REDUCED_COLOR_8)",
                    lambda a: cv2.imdecode(a, cv2.IMREAD_REDUCED_COLOR_8), arrs)
    r["gray"] = bench("full decode GRAYSCALE",
                      lambda a: cv2.imdecode(a, cv2.IMREAD_GRAYSCALE), arrs)

    print("\nSPEEDUP vs full decode")
    base = r["full"][0]
    for k in ("r2", "r4", "r8", "gray"):
        print("  %-6s %6.2fx   (%s)" % (k, base / r[k][0], r[k][1]))

    print("\nNOTE: the network needs 640x640. Any decode whose output is still")
    print("      >= 640 in both dimensions loses NO information the pipeline")
    print("      would have kept anyway.")
    for k in ("full", "r2", "r4", "r8"):
        s = r[k][1]
        ok = "OK  - still >=640" if (s and min(s[0], s[1]) >= 640) else "TOO SMALL - would upscale"
        print("  %-6s %-16s %s" % (k, str(s[:2]) if s else "-", ok))
    return 0


if __name__ == "__main__":
    sys.exit(main())
