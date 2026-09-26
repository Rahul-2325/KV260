#!/usr/bin/env python3
"""
dump_graph_plan.py -- run ON THE BOARD.

Prints everything needed to hand-write the DPU -> LCAM-IP -> DPU pipeline:
  * every subgraph in topological order, DPU or CPU
  * each subgraph's input and output tensor names (the plumbing)
  * for CPU subgraphs, every op with its type and the attributes a numpy
    re-implementation needs (fix_point, axis, order, ...)

This is the map for the runner. GraphRunner does this dispatch internally
and gives no hook to redirect one op to a custom CU, which is exactly why
the loop has to be written by hand.
"""
import xir

XMODEL = "/home/root/lcam_hybrid.xmodel"

# attributes a numpy re-implementation actually needs
WANTED = ("fix_point", "bit_width", "round_mode", "axis", "order",
          "shape", "data_type", "if_signed")


def short(n, keep=46):
    """Op/tensor names are enormous; keep the distinguishing tail."""
    return n if len(n) <= keep else "..." + n[-(keep - 3):]


def main():
    g = xir.Graph.deserialize(XMODEL)
    root = g.get_root_subgraph()
    subs = root.toposort_child_subgraph()

    print("total subgraphs: %d\n" % len(subs))

    for i, s in enumerate(subs):
        dev = s.get_attr("device").upper() if s.has_attr("device") else "?"
        ops = list(s.get_ops())

        # subgraph inputs = tensors produced outside this subgraph
        inside = set()
        for op in ops:
            inside.add(op.get_output_tensor().name)
        ins = []
        for op in ops:
            for t in op.get_input_tensors():
                if t.name not in inside and t.name not in ins:
                    ins.append(t.name)

        outs = [t.name for t in s.get_output_tensors()]

        print("=" * 74)
        print("[%02d] %-3s  ops=%d" % (i, dev, len(ops)))
        for n in ins:
            print("      IN   %s" % short(n))
        for n in outs:
            print("      OUT  %s" % short(n))

        if dev == "DPU":
            continue

        # CPU subgraph: this is what has to be reimplemented
        print("      ---- ops to reimplement ----")
        for op in ops:
            t = op.get_output_tensor()
            attrs = []
            for a in WANTED:
                if op.has_attr(a):
                    attrs.append("%s=%s" % (a, op.get_attr(a)))
                elif t.has_attr(a):
                    attrs.append("t.%s=%s" % (a, t.get_attr(a)))
            srcs = [short(x.name, 30) for x in op.get_input_tensors()]
            print("      %-14s out=%-20s %s"
                  % (op.get_type(), str(list(t.dims)), " ".join(attrs)))
            for sname in srcs:
                print("          <- %s" % sname)


if __name__ == "__main__":
    main()
