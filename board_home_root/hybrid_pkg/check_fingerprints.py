#!/usr/bin/env python3
"""
check_fingerprints.py -- run ON THE BOARD.

Lists the DPU fingerprint every local xmodel was compiled for, so we can
see at a glance which (if any) matches the DPU actually in the PL.

Reads metadata only -- builds no runner, touches no register. Safe.

The hardware fingerprint comes from `xdputil query`, which for the
kv260-hybrid2 build reports 0x101000012010407.
"""
import glob
import xir

HW = 0x101000012010407


def fingerprints(path):
    g = xir.Graph.deserialize(path)
    root = g.get_root_subgraph()
    fps, n = set(), 0
    if not root.is_leaf:
        for s in root.toposort_child_subgraph():
            if s.has_attr("device") and s.get_attr("device").upper() == "DPU":
                n += 1
                for a in ("dpu_fingerprint", "fingerprint"):
                    if s.has_attr(a):
                        fps.add(s.get_attr(a))
    return n, fps


def main():
    print("HARDWARE dpu_fingerprint = %s\n" % hex(HW))
    paths = sorted(glob.glob("/home/root/*.xmodel"))
    for p in paths:
        name = p.split("/")[-1]
        try:
            n, fps = fingerprints(p)
        except Exception as e:
            print("%-26s ERROR %s" % (name, e))
            continue
        shown = ",".join(hex(v) if isinstance(v, int) else str(v) for v in sorted(fps))
        match = "MATCH" if HW in fps else "no"
        print("%-26s dpu_subgraphs=%-3d %-20s %s" % (name, n, shown or "-", match))


if __name__ == "__main__":
    main()
