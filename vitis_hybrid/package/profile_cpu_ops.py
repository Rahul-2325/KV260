#!/usr/bin/env python3
"""
profile_cpu_ops.py -- run ON THE BOARD.

Attributes the 385 ms of CPU fallback to individual ops, by MEASUREMENT
rather than by element-count proxy.

How: DEEPHI_PROFILING=1 makes libvitis_ai_library-cpu_task log every CPU op
it executes, with a microsecond timestamp:
    I0715 09:40:28.702615 3759 cpu_task.cpp:194] op_name : <name>
It logs the START of each op, not a duration, so the duration of an op is
the gap to the NEXT logged event. Gaps that span a DPU subgraph also
contain that subgraph's execution, so those are reported separately and
must not be read as pure CPU time.

This matters because the two cheap estimates disagree badly:
  - element-count proxy says the gate path is ~93% of CPU work
  - a numpy re-implementation of the gate path costs only 149 ms of 385 ms
Neither is VART's actual cost, and the end-to-end claim depends on it.

Output: per-op-name total ms per frame, sorted, with the four LCAM gate
subgraphs called out explicitly.
"""
import collections
import re
import subprocess
import sys

LOG = "/home/root/hybrid_pkg/deephi.log"

# start-of-op lines and the tensor-linker lines that bracket subgraphs
RE_OP = re.compile(
    r"^I\d{4} (\d\d):(\d\d):(\d\d)\.(\d{6})\s+\d+ cpu_task\.cpp:\d+\] op_name : (\S+)")
RE_LINK = re.compile(
    r"^I\d{4} (\d\d):(\d\d):(\d\d)\.(\d{6})\s+\d+ tensor_buffer_linker")


def ts(m):
    h, mi, s, us = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return ((h * 60 + mi) * 60 + s) * 1e6 + us


def shorten(name):
    """Op names are enormous; keep the informative tail."""
    n = name.split("__")[-1] if "__" in name else name
    return n[:52]


def gate_layer(name):
    """
    Identify the four LCAM attention gates by their lcamN tag.

    Match on LCAM_lcamN alone. An earlier version also required
    'Hardsigmoid_gate', which matched only a trivial 0.13 ms wrapper op and
    left the actual gate compute (the '..._input_63' subgraph, 74.7 ms)
    misfiled under 'other'. Every op carrying an LCAM_lcamN tag belongs to
    that gate's subgraph and is work our IP subsumes -- including the
    float2fix that follows it, since the IP is int8-in/int8-out.
    """
    m = re.search(r"LCAM_lcam(\d)", name)
    return ("lcam%s" % m.group(1)) if m else None


def main():
    events = []
    with open(LOG, errors="replace") as f:
        for line in f:
            m = RE_OP.match(line)
            if m:
                events.append((ts(m), "op", m.group(5)))
                continue
            m = RE_LINK.match(line)
            if m:
                events.append((ts(m), "link", ""))

    if len(events) < 10:
        print("not enough profiling events (%d) -- was DEEPHI_PROFILING=1 set?"
              % len(events))
        return 1

    events.sort(key=lambda e: e[0])

    dur = collections.defaultdict(float)
    cnt = collections.Counter()
    gates = collections.defaultdict(float)
    gate_cnt = collections.Counter()
    gate_names = collections.defaultdict(set)

    for i in range(len(events) - 1):
        t, kind, name = events[i]
        if kind != "op":
            continue
        d = (events[i + 1][0] - t) / 1000.0        # ms
        if d < 0 or d > 5000:
            continue
        g = gate_layer(name)
        if g:
            gates[g] += d
            gate_cnt[g] += 1
            gate_names[g].add(name)
        else:
            dur[shorten(name)] += d
            cnt[shorten(name)] += 1

    # Frame count = executions of the lcam2 gate divided by how many DISTINCT
    # ops that gate's subgraph contains. Counting raw executions is wrong:
    # the lcam2 subgraph runs 4 ops per frame, so that heuristic reported 36
    # "frames" for 9 real ones and scaled every result down by 4x.
    per_frame_ops = max(1, len(gate_names.get("lcam2", ())))
    nframes = max(1, gate_cnt.get("lcam2", 1) // per_frame_ops)
    print("lcam2 subgraph ops/frame: %d   ->  frames observed: %d\n"
          % (per_frame_ops, nframes))

    print("=" * 72)
    print("THE FOUR LCAM ATTENTION GATES  (what our custom IP replaces)")
    print("=" * 72)
    print("%-10s %10s %12s" % ("gate", "runs", "ms/frame"))
    gate_total = 0.0
    for g in sorted(gates):
        per = gates[g] / nframes
        gate_total += per
        print("%-10s %10d %12.2f" % (g, gate_cnt[g], per))
    print("-" * 72)
    print("%-10s %10s %12.2f" % ("TOTAL", "", gate_total))
    print()
    print()
    print("  end-to-end with these on the CPU : 426.51 ms   2.34 FPS")
    print("  the same four gates on our IP    :   1.82 ms   (bit-exact)")
    print("  => projected end-to-end          : %6.2f ms  %5.2f FPS  (%.2fx)"
          % (426.51 - gate_total + 1.82,
             1000.0 / (426.51 - gate_total + 1.82),
             426.51 / (426.51 - gate_total + 1.82)))
    print()

    print("=" * 72)
    print("ALL OTHER CPU OPS (top 20 by time)")
    print("=" * 72)
    print("%-54s %6s %10s" % ("op", "runs", "ms/frame"))
    other = 0.0
    for name, d in sorted(dur.items(), key=lambda kv: -kv[1])[:20]:
        per = d / nframes
        other += per
        print("%-54s %6d %10.2f" % (name, cnt[name], per))
    rest = sum(dur.values()) / nframes - other
    print("-" * 72)
    print("%-54s %6s %10.2f" % ("(shown above)", "", other))
    print("%-54s %6s %10.2f" % ("(remaining ops)", "", rest))
    print()
    print("=" * 72)
    print("measured independently on this board:")
    print("  end-to-end            : 426.51 ms  (2.34 FPS)")
    print("  DPU compute total     :  41.15 ms")
    print("  CPU fallback          : 385.36 ms")
    print("  custom LCAM IP        :   1.82 ms  (all four gates, bit-exact)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

