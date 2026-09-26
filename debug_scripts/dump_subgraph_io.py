#!/usr/bin/env python3
"""
dump_subgraph_io.py
Dumps every subgraph's input/output tensors (name, shape, fix_point) in
topological order.

This is the wiring diagram for the hybrid runner: to execute the model
manually we must know exactly which tensor each subgraph consumes and
produces, so DPU results can be fed to the custom LCAM IP and back.

GraphRunner does this automatically but gives us no way to substitute the
custom IP for the CPU attention subgraphs -- hence the manual pipeline.
"""
import argparse
import xir


def ops_of(sg):
    for meth in ('get_ops', 'toposort_child_subgraph', 'get_children'):
        if not hasattr(sg, meth):
            continue
        try:
            items = getattr(sg, meth)()
        except Exception:
            continue
        if items and hasattr(list(items)[0], 'get_type'):
            return list(items)
    return []


def tinfo(t):
    fp = None
    for k in ('fix_point', 'fix_pos'):
        try:
            if t.has_attr(k):
                fp = t.get_attr(k); break
        except Exception:
            pass
    return "%-58s %-22s fp=%s" % (t.name[:58], str(list(t.dims)), fp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xmodel', default='/home/root/lcam_v5_2p5.xmodel')
    args = ap.parse_args()

    g = xir.Graph.deserialize(args.xmodel)
    subs = g.get_root_subgraph().toposort_child_subgraph()
    print("=" * 96)
    print("SUBGRAPH I/O MAP  (%s)" % args.xmodel)
    print("=" * 96)

    for i, sg in enumerate(subs):
        dev = sg.get_attr('device') if sg.has_attr('device') else 'NONE'
        ops = ops_of(sg)
        types = sorted({o.get_type() for o in ops})
        # mark the LCAM attention gates: CPU subgraphs containing 'mul'
        tag = ""
        if dev == 'CPU' and 'mul' in types:
            tag = "   <<< LCAM ATTENTION GATE -> custom IP"
        print("\n[%2d] device=%-5s ops=%-3d %s%s" % (i, dev, len(ops), types, tag))
        try:
            for t in sg.get_input_tensors():
                print("      IN   %s" % tinfo(t))
        except Exception as e:
            print("      (input query failed: %s)" % e)
        try:
            for t in sg.get_output_tensors():
                print("      OUT  %s" % tinfo(t))
        except Exception as e:
            print("      (output query failed: %s)" % e)

    print()
    print("=" * 96)
    print("GRAPH-LEVEL")
    print("=" * 96)
    try:
        for t in g.get_input_tensors():
            print("  MODEL IN  : %s" % tinfo(t))
    except Exception:
        pass
    try:
        for t in g.get_output_tensors():
            print("  MODEL OUT : %s" % tinfo(t))
    except Exception:
        pass


if __name__ == '__main__':
    main()
