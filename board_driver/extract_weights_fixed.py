#!/usr/bin/env python3
"""
extract_weights_fixed.py
Correctly extracts ALL const-fix tensors from lcam_v5.xmodel.

Replaces the original extraction, which failed on every tensor with
  'xir.Tensor' object has no attribute 'to_numpy'
leaving weight_map.json with 390/390 null file entries (and only a
partial, separately-produced set of 211 .npy files with 5 genuinely
missing tensors). See PROJECT_HISTORY.md §27.

Correct API: the raw bytes live on the OP as attribute "data";
shape/dtype come from op.get_output_tensor().

Outputs:
  weights_v2/data/<sanitized>.npy   one file per const tensor
  weights_v2/weight_index.json      full_tensor_name -> {file, shape, dtype, fp}
"""
import os, json, re, hashlib
import numpy as np
import xir

XMODEL = '/home/root/lcam_v5.xmodel'
OUTDIR = '/home/root/weights_v2'
DATADIR = os.path.join(OUTDIR, 'data')

DTYPE_MAP = {
    'xint8': np.int8, 'int8': np.int8, 'xuint8': np.uint8, 'uint8': np.uint8,
    'xint16': np.int16, 'int16': np.int16,
    'xint32': np.int32, 'int32': np.int32,
    'float32': np.float32, 'float': np.float32,
    'xfloat32': np.float32,
}

def sanitize(name):
    """Filesystem-safe, collision-free filename for a long tensor name."""
    base = re.sub(r'[^A-Za-z0-9_.-]', '_', name)
    if len(base) > 120:
        h = hashlib.md5(name.encode()).hexdigest()[:8]
        base = base[:110] + '_' + h
    return base + '.npy'

def main():
    os.makedirs(DATADIR, exist_ok=True)
    g = xir.Graph.deserialize(XMODEL)
    ops = g.get_ops()
    consts = [o for o in ops if 'const' in o.get_type()]
    print("const ops found: %d" % len(consts))

    index = {}
    ok = fail = 0
    for o in consts:
        name = o.get_name()
        try:
            t = o.get_output_tensor()
            dims = list(t.dims)
            dtype_str = str(t.dtype)
            npdt = DTYPE_MAP.get(dtype_str)
            if npdt is None:
                # fall back: infer width from byte count
                nbytes = t.get_data_size()
                n = int(np.prod(dims)) if dims else 1
                npdt = {1: np.int8, 2: np.int16, 4: np.int32}.get(
                    max(1, nbytes // max(1, n)), np.int8)
                print("  [warn] unknown dtype %r for %s -> using %s"
                      % (dtype_str, name[:40], np.dtype(npdt).name))

            raw = o.get_attr('data')
            arr = np.frombuffer(bytes(raw), dtype=npdt)
            if dims and arr.size == int(np.prod(dims)):
                arr = arr.reshape(dims)
            elif dims:
                print("  [warn] size mismatch %s: got %d want %d (kept flat)"
                      % (name[:40], arr.size, int(np.prod(dims))))

            fn = sanitize(name)
            np.save(os.path.join(DATADIR, fn), arr)

            fp = None
            for key in ('fix_point', 'fix_pos'):
                if t.has_attr(key):
                    fp = t.get_attr(key); break
            index[name] = {'file': fn, 'shape': list(arr.shape),
                           'dtype': np.dtype(npdt).name, 'fp': fp}
            ok += 1
        except Exception as e:
            fail += 1
            print("  [FAIL] %s: %s" % (name[:60], e))

    with open(os.path.join(OUTDIR, 'weight_index.json'), 'w') as f:
        json.dump(index, f, indent=1)

    print("\nextracted OK : %d" % ok)
    print("failed       : %d" % fail)
    print("index written: %s/weight_index.json" % OUTDIR)

if __name__ == '__main__':
    main()
