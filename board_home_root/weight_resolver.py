#!/usr/bin/env python3
"""
weight_resolver.py
Resolves each layer's constant tensors (bias / weight) to the extracted
.npy files, working around two defects in the original extraction:

  1. weight_map.json is USELESS -- all 390 entries have "file": null
     because the extractor called the non-existent xir.Tensor.to_numpy().
     We ignore it entirely and resolve by name instead.

  2. Both the plan's tensor names and the .npy filenames are TRUNCATED
     (differently), so matching must be done by SUFFIX -- and suffix
     matching alone is ambiguous: it silently picks the wrong file for 6
     layers. Every match is therefore SHAPE-VERIFIED against the plan.

Tensor slot convention in layer_plan.json (verified against the graph):
    conv2d-fix / depthwise-fix : input_0 = bias, input_1 = activation,
                                 input_2 = weight
    pool/eltwise/hard-sigmoid  : input_0 (and input_1) are ACTIVATIONS,
                                 no constants to resolve.
"""
import json, os
import numpy as np

DATA_DIR  = '/home/root/weights/data'
PLAN_PATH = '/home/root/weights/layer_plan.json'

CONST_OPS = ('conv2d-fix', 'depthwise-fix')
BIAS_SLOT, WEIGHT_SLOT = 'input_0', 'input_2'


class WeightResolver:
    def __init__(self, data_dir=DATA_DIR, plan_path=PLAN_PATH):
        self.data_dir = data_dir
        self.plan = json.load(open(plan_path))
        self.files = [f for f in sorted(os.listdir(data_dir)) if f.endswith('.npy')]
        self._shape_cache = {}
        self.unresolved = []

    def _shape(self, fname):
        if fname not in self._shape_cache:
            self._shape_cache[fname] = list(
                np.load(os.path.join(self.data_dir, fname), mmap_mode='r').shape)
        return self._shape_cache[fname]

    def _candidates(self, name):
        n = name[:-4] if name.endswith('_fix') else name
        c = [f for f in self.files if f[:-4].endswith(n)]
        if not c:
            c = [f for f in self.files if f[:-4].endswith(n.lstrip('_'))]
        return c

    def resolve(self, name, want_shape):
        """Return filename whose SHAPE matches want_shape, or None."""
        if not name:
            return None
        cands = self._candidates(name)
        exact = [f for f in cands if self._shape(f) == list(want_shape)]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            return exact[0]          # identical shape+suffix: any is fine
        # suffix matched but shape disagreed -> fall back to global shape search
        glob = [f for f in self.files if self._shape(f) == list(want_shape)]
        if len(glob) == 1:
            return glob[0]
        return None

    def load(self, fname):
        return np.load(os.path.join(self.data_dir, fname))

    def build(self):
        """Returns {layer_id: {'bias': arr|None, 'weight': arr|None}}."""
        table, missing = {}, []
        for L in self.plan:
            if L['op_type'] not in CONST_OPS:
                continue
            entry = {}
            for slot, key in ((BIAS_SLOT, 'bias'), (WEIGHT_SLOT, 'weight')):
                nm, sh = L.get(slot + '_name'), L.get(slot + '_shape')
                if nm is None:
                    entry[key] = None
                    continue
                f = self.resolve(nm, sh)
                if f is None:
                    entry[key] = None
                    missing.append((L['id'], L['op_type'], key, nm, sh))
                else:
                    entry[key] = f
            entry['fp'] = {
                'in':   L.get('input_1_fp'),
                'w':    L.get('input_2_fp'),
                'bias': L.get('input_0_fp'),
                'out':  L.get('fp_out'),
            }
            table[L['id']] = entry
        self.unresolved = missing
        return table


if __name__ == '__main__':
    r = WeightResolver()
    tbl = r.build()
    n_layers = len(tbl)
    full = sum(1 for v in tbl.values()
               if v['weight'] is not None and v['bias'] is not None)
    print("conv/depthwise layers      : %d" % n_layers)
    print("  fully resolved           : %d" % full)
    print("  with a missing tensor    : %d" % (n_layers - full))
    print("  unresolved tensor count  : %d" % len(r.unresolved))
    if r.unresolved:
        print("\nUNRESOLVED:")
        for lid, op, key, nm, sh in r.unresolved:
            print("  L%-4d %-14s %-7s %-44s %s" % (lid, op, key, nm[:44], sh))
    # verify every resolved file's shape really matches the plan
    bad = 0
    for L in r.plan:
        if L['op_type'] not in CONST_OPS:
            continue
        e = tbl[L['id']]
        for slot, key in ((BIAS_SLOT, 'bias'), (WEIGHT_SLOT, 'weight')):
            if e[key] is None:
                continue
            if r._shape(e[key]) != list(L[slot + '_shape']):
                bad += 1
                print("  SHAPE MISMATCH L%d %s" % (L['id'], key))
    print("\nshape mismatches among resolved: %d" % bad)
