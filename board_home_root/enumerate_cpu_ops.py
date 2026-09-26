#!/usr/bin/env python3
"""
enumerate_cpu_ops.py
Lists every op inside every CPU subgraph, with shapes, and maps each to
the custom IP that could replace it.

Purpose: the DPU/CPU profile showed 90.5% of frame time is CPU fallback.
This tells us WHICH ops that is, and therefore how much of it our 9 custom
IPs can actually take over -- the difference between a real speedup claim
and an over-claim.
"""
import argparse, collections
import xir

# op type -> custom IP that implements it
IP_MAP = {
    'depthwise-fix':    'lcam_attention_gate (gate multiply)',
    'pool-fix':         'pool_engine',
    'eltwise-fix':      'eltwise_add',
    'hard-sigmoid-fix': 'head_sigmoid',
    'conv2d-fix':       'conv2d_engine',
    'concat-fix':       'head_concat_reshape',
    'reshape-fix':      'head_concat_reshape (reshape path)',
    'sigmoid':          'head_sigmoid (approx)',
    'transpose':        'head_transpose',
    'upsample':         'upsample_engine',
}


def ops_of(sg):
    """Return the leaf ops of a subgraph, tolerating API differences."""
    for meth in ('get_ops', 'toposort_child_subgraph', 'get_children'):
        if not hasattr(sg, meth):
            continue
        try:
            items = getattr(sg, meth)()
        except Exception:
            continue
        if not items:
            continue
        # ops have get_type(); subgraphs don't
        if hasattr(list(items)[0], 'get_type'):
            return list(items)
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xmodel', default='/home/root/lcam_v5_2p5.xmodel')
    args = ap.parse_args()

    g = xir.Graph.deserialize(args.xmodel)
    subs = g.get_root_subgraph().toposort_child_subgraph()

    print("=" * 74)
    print("CPU SUBGRAPH CONTENTS  (%s)" % args.xmodel)
    print("=" * 74)

    tally = collections.Counter()
    elems = collections.Counter()
    covered = collections.Counter()

    for i, s in enumerate(subs):
        dev = s.get_attr('device') if s.has_attr('device') else 'NONE'
        if dev != 'CPU':
            continue
        ops = ops_of(s)
        print("\n[subgraph %d]  %d op(s)" % (i, len(ops)))
        for o in ops:
            t = o.get_type()
            try:
                dims = list(o.get_output_tensor().dims)
                n = 1
                for d in dims:
                    n *= d
            except Exception:
                dims, n = [], 0
            tally[t] += 1
            elems[t] += n
            ip = IP_MAP.get(t)
            if ip:
                covered[t] += n
            print("   %-18s out=%-22s %10s elem   -> %s"
                  % (t, str(dims), "{:,}".format(n), ip or "** no custom IP **"))

    print()
    print("=" * 74)
    print("SUMMARY BY OP TYPE (across all CPU subgraphs)")
    print("=" * 74)
    print("%-20s %6s %14s  %s" % ("op type", "count", "elements", "custom IP"))
    print("-" * 74)
    for t, c in tally.most_common():
        print("%-20s %6d %14s  %s"
              % (t, c, "{:,}".format(elems[t]), IP_MAP.get(t, "** NONE **")))

    tot = sum(elems.values())
    cov = sum(covered.values())
    print("-" * 74)
    print("total CPU-side elements       : %s" % "{:,}".format(tot))
    print("covered by an existing IP     : %s (%.1f%%)"
          % ("{:,}".format(cov), 100.0 * cov / tot if tot else 0))
    print("NOT covered (stay on CPU)     : %s (%.1f%%)"
          % ("{:,}".format(tot - cov), 100.0 * (tot - cov) / tot if tot else 0))
    print("=" * 74)
    print("NOTE: element counts are a PROXY for work, not measured time.")
    print("      Use them to prioritise, not as a speedup claim.")


if __name__ == '__main__':
    main()
