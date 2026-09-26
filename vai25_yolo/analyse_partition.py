#!/usr/bin/env python3
"""
Why does this xmodel get split across DPU and CPU?

Dumps every subgraph in execution order, and for each CPU subgraph lists the
exact ops (type + output shape) that the compiler refused to put on the DPU.
That tells you precisely which layer to change if you want a single-subgraph,
DPU-resident model.

  python3 analyse_partition.py <xmodel>
"""
import sys
from collections import Counter
import xir


def main():
    path = sys.argv[1]
    g = xir.Graph.deserialize(path)
    subs = g.get_root_subgraph().toposort_child_subgraph()

    print("=" * 74)
    print("MODEL: %s" % path.split("/")[-1])
    print("=" * 74)

    dev_count = Counter()
    for s in subs:
        dev_count[s.get_attr("device").upper() if s.has_attr("device") else "?"] += 1
    print("subgraphs: %d total  ->  %s" % (len(subs), dict(dev_count)))
    print("")

    cpu_ops = Counter()
    cpu_op_shapes = {}

    for i, s in enumerate(subs):
        dev = s.get_attr("device").upper() if s.has_attr("device") else "?"
        try:
            outs = list(s.get_output_tensors())
            oshape = tuple(outs[0].dims) if outs else None
        except Exception:
            oshape = None

        ops = list(s.get_ops())
        types = Counter(o.get_type() for o in ops)

        marker = "  " if dev == "DPU" else ">>"
        print("%s [%02d] %-4s  %-22s  %d ops" %
              (marker, i, dev, str(oshape), len(ops)))

        if dev != "DPU":
            for o in ops:
                t = o.get_type()
                cpu_ops[t] += 1
                try:
                    sh = tuple(o.get_output_tensor().dims)
                except Exception:
                    sh = None
                cpu_op_shapes.setdefault(t, set()).add(sh)
            # show the ops of this CPU subgraph
            for t, c in types.most_common():
                print("         %-22s x%d" % (t, c))

    print("")
    print("=" * 74)
    print("EVERY OP THE COMPILER PUT ON THE CPU")
    print("=" * 74)
    print("%-24s %6s   %s" % ("op type", "count", "output shapes seen"))
    print("-" * 74)
    total_elems = 0
    for t, c in cpu_ops.most_common():
        shapes = sorted(x for x in cpu_op_shapes[t] if x)
        shown = ", ".join(str(x) for x in shapes[:3])
        if len(shapes) > 3:
            shown += ", ..."
        print("%-24s %6d   %s" % (t, c, shown))
        for sh in cpu_op_shapes[t]:
            if sh:
                n = 1
                for d in sh:
                    n *= d
                total_elems += n
    print("-" * 74)
    print("distinct CPU op types : %d" % len(cpu_ops))
    print("total CPU ops         : %d" % sum(cpu_ops.values()))
    print("")
    print("NOTE: DPUCZDX8G supports hard-sigmoid, hard-swish and element-wise")
    print("      multiply of MATCHING shapes. A multiply whose two inputs have")
    print("      DIFFERENT shapes (broadcast, e.g. [1,H,W,1] x [1,H,W,C]) is a")
    print("      common reason an attention gate is pushed to the CPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
