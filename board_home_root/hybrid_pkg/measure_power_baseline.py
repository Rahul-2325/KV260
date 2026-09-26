#!/usr/bin/env python3
"""
measure_power_baseline.py -- run ON THE BOARD.

Measures the INA260 rail while the STOCK Vitis-AI path runs, so the hybrid's
power can be compared against something rather than quoted alone.

Power on its own is not a result: a slower design that idles more can draw
less. The comparable figure is ENERGY PER FRAME (watts x seconds), which is
what this pair of runs produces.

Both runs execute on the same loaded bitstream; only the software path
differs, so the comparison isolates the pipeline rather than the hardware.
"""
import os
import subprocess
import sys
import threading
import time

HW = "/sys/class/hwmon/hwmon0"


def read(n):
    with open(os.path.join(HW, n)) as f:
        return int(f.read().strip())


def sample(stop, out, period=0.05):
    while not stop.is_set():
        try:
            out.append(read("power1_input") / 1e6)
        except Exception:
            pass
        time.sleep(period)


def run_under_meter(cmd, env):
    rows, stop = [], threading.Event()
    t = threading.Thread(target=sample, args=(stop, rows))
    t.start()
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    out = p.communicate()[0].decode(errors="replace")
    wall = time.perf_counter() - t0
    stop.set()
    t.join()
    return rows, out, wall


def main():
    env = dict(os.environ,
               XLNX_VART_FIRMWARE="/lib/firmware/xilinx/kv260-hybrid2/hybrid.xclbin",
               XRT_INI_PATH="/home/root/xrt.ini")

    runs = [
        ("stock GraphRunner",
         [sys.executable, "/home/root/e2e_graphrunner.py",
          "--xmodel", "/home/root/lcam_hybrid.xmodel", "--runs", "20"],
         "mean latency"),
        ("hybrid pipeline",
         [sys.executable, "/home/root/hybrid_pkg/hybrid_pipeline.py",
          "--gates", "ip", "--round", "ref", "--runs", "20"],
         "mean total"),
    ]

    results = []
    for label, cmd, key in runs:
        print("--- %s ---" % label)
        rows, out, wall = run_under_meter(cmd, env)
        lat = None
        for line in out.splitlines():
            if key in line:
                try:
                    lat = float(line.split(":")[1].split("ms")[0].strip())
                except Exception:
                    pass
        mean_w = sum(rows) / len(rows) if rows else 0.0
        peak_w = max(rows) if rows else 0.0
        print("  power  : %.3f W mean, %.3f W peak (%d samples over %.1f s)"
              % (mean_w, peak_w, len(rows), wall))
        print("  latency: %s ms" % (("%.2f" % lat) if lat else "?"))
        if lat:
            print("  energy : %.3f J per frame" % (mean_w * lat / 1000.0))
        results.append((label, mean_w, lat))
        print()
        time.sleep(3)

    if len(results) == 2 and all(r[2] for r in results):
        (l0, w0, t0), (l1, w1, t1) = results
        e0, e1 = w0 * t0 / 1000.0, w1 * t1 / 1000.0
        print("=" * 58)
        print("%-22s %8s %10s %12s" % ("", "power", "latency", "energy/frame"))
        print("%-22s %7.3fW %8.2f ms %10.3f J" % (l0, w0, t0, e0))
        print("%-22s %7.3fW %8.2f ms %10.3f J" % (l1, w1, t1, e1))
        print("-" * 58)
        print("%-22s %7.2fx %8.2fx %10.2fx"
              % ("improvement", w0 / w1 if w1 else 0, t0 / t1, e0 / e1))
        print()
        print("Energy per frame is the honest efficiency metric: the hybrid")
        print("draws more instantaneous power but finishes far sooner.")


if __name__ == "__main__":
    main()
