#!/usr/bin/env python3
"""
measure_power.py -- run ON THE BOARD.

MEASURED board power, as opposed to Vivado's report_power estimate.

Vivado's number is vectorless: it infers switching activity from the netlist
and default assumptions, and reports the PL only. It says nothing about the
processing system, DDR, peripherals, or what the design actually toggles on
real data. The KV260 SOM carries an INA260 on the main input rail, exposed at
/sys/class/hwmon/hwmon0 (name "ina260_u14"), so the true figure is readable:

    power1_input   microwatts
    curr1_input    milliamps
    in1_input      millivolts

Method: sample the rail while the board is idle, then again while the hybrid
pipeline runs continuously, and report the difference. The delta is the cost
of running the network; the absolute value is what the board actually draws.

Sampling runs in a thread so the load process is timed unperturbed.
"""
import argparse
import os
import subprocess
import sys
import threading
import time

HW = "/sys/class/hwmon/hwmon0"


def read(node):
    with open(os.path.join(HW, node)) as f:
        return int(f.read().strip())


def sample(stop, out, period=0.05):
    while not stop.is_set():
        try:
            out.append((read("power1_input") / 1e6,      # W
                        read("curr1_input") / 1e3,       # A
                        read("in1_input") / 1e3))        # V
        except Exception:
            pass
        time.sleep(period)


def stats(rows, label):
    if not rows:
        print("  %-22s no samples" % label)
        return None
    p = sorted(r[0] for r in rows)
    n = len(p)
    mean = sum(p) / n
    print("  %-22s %6.3f W mean   %6.3f min   %6.3f max   %6.3f median   (%d samples)"
          % (label, mean, p[0], p[-1], p[n // 2], n))
    print("  %-22s %6.3f A         %6.3f V"
          % ("", sum(r[1] for r in rows) / n, sum(r[2] for r in rows) / n))
    return mean


def collect(seconds, worker=None):
    rows, stop = [], threading.Event()
    t = threading.Thread(target=sample, args=(stop, rows))
    t.start()
    if worker is None:
        time.sleep(seconds)
    else:
        worker()
    stop.set()
    t.join()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idle", type=float, default=8.0)
    ap.add_argument("--frames", type=int, default=60)
    args = ap.parse_args()

    print("sensor: %s" % open(os.path.join(HW, "name")).read().strip())
    print("(INA260 on the SOM input rail -- whole-module power, PS + PL + DDR)\n")

    print("--- idle (design loaded, nothing running) ---")
    time.sleep(1.0)
    idle = stats(collect(args.idle), "idle")

    print("\n--- running the hybrid pipeline continuously ---")
    env = dict(os.environ,
               XLNX_VART_FIRMWARE="/lib/firmware/xilinx/kv260-hybrid2/hybrid.xclbin",
               XRT_INI_PATH="/home/root/xrt.ini")
    proc = {}

    def run():
        proc["p"] = subprocess.Popen(
            [sys.executable, "/home/root/hybrid_pkg/hybrid_pipeline.py",
             "--gates", "ip", "--round", "ref", "--runs", str(args.frames)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        proc["out"] = proc["p"].communicate()[0].decode(errors="replace")

    load = stats(collect(0, run), "under load")

    for line in proc.get("out", "").splitlines():
        if "mean total" in line or "speedup" in line:
            print("   %s" % line.strip())

    if idle and load:
        print("\n--- result ---")
        print("  idle                 : %6.3f W" % idle)
        print("  running              : %6.3f W" % load)
        print("  delta (the workload) : %6.3f W" % (load - idle))
        print()
        print("  Vivado report_power for the PL is an ESTIMATE and covers the")
        print("  PL only; this is the measured whole-module draw. The two are")
        print("  not comparable line for line -- quote this one for the board.")


if __name__ == "__main__":
    main()
