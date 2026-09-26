import xir, sys, os
for p in sys.argv[1:]:
    try:
        g = xir.Graph.deserialize(p)
        subs = g.get_root_subgraph().toposort_child_subgraph()
        fps = set()
        for s in subs:
            if s.has_attr('dpu_fingerprint'):
                fps.add(hex(s.get_attr('dpu_fingerprint')))
        print('%-46s %s' % (os.path.basename(p), sorted(fps)))
    except Exception as e:
        print('%-46s ERROR %s' % (os.path.basename(p), e))
