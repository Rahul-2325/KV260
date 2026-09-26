#!/usr/bin/env python3
"""
verify_lcam_int_math.py
Proves the OPTIMISED integer arithmetic in lcam_attention_gate_opt.cpp is
BIT-IDENTICAL to the original float implementation, before spending a
40-minute synthesis run on it.

original (lcam_attention_gate.cpp):
    feat_f  = feat   * 2^-fp_feat
    w_f     = weight * 2^-fp_weight
    q       = roundf(feat_f * w_f * 2^fp_out)  then clamp to int8

optimised:
    shift = fp_feat + fp_weight - fp_out
    mag   = |feat*weight|;  r = (mag + (1<<(shift-1))) >> shift
    q     = sign(feat*weight) * r            then clamp to int8

The magnitude-then-sign form matters: roundf() rounds HALF AWAY FROM
ZERO, while a bare arithmetic shift rounds half toward -infinity, so a
naive "(prod + bias) >> shift" disagrees on negative ties.

Exhaustive over the full int8 x int8 domain for every real layer config.
"""
import numpy as np

# (name, fp_feat, fp_weight, fp_out) for the four real gate layers
CONFIGS = [
    ("L23 160x160x64", 4, 7, 5),
    ("L41  80x80x128", 5, 7, 5),
    ("L53  40x40x256", 5, 7, 5),
    ("L83  20x20x512", 5, 7, 6),
]


def original(feat, weight, fp_feat, fp_weight, fp_out):
    """float path, exactly as the current HLS does it"""
    f = np.float32(feat) * np.float32(1.0 / (1 << fp_feat))
    w = np.float32(weight) * np.float32(1.0 / (1 << fp_weight))
    scaled = np.float32(f * w) * np.float32(1 << fp_out)
    # C roundf = half away from zero
    q = np.floor(np.abs(scaled) + 0.5) * np.sign(scaled)
    return int(np.clip(q, -128, 127))


def optimised(feat, weight, fp_feat, fp_weight, fp_out):
    """integer path, exactly as lcam_attention_gate_opt.cpp does it"""
    shift = fp_feat + fp_weight - fp_out
    prod = int(feat) * int(weight)
    if shift > 0:
        bias = 1 << (shift - 1)
        mag = -prod if prod < 0 else prod
        r = (mag + bias) >> shift
        q = -r if prod < 0 else r
    else:
        q = prod << (-shift)
    return max(-128, min(127, q))


def naive(feat, weight, fp_feat, fp_weight, fp_out):
    """the WRONG version (round-half-up), kept to show why sign matters"""
    shift = fp_feat + fp_weight - fp_out
    prod = int(feat) * int(weight)
    q = (prod + (1 << (shift - 1))) >> shift if shift > 0 else prod << (-shift)
    return max(-128, min(127, q))


print("=" * 68)
print("EXHAUSTIVE int8 x int8 CHECK  (65,536 combinations per config)")
print("=" * 68)
all_ok = True
for name, fpf, fpw, fpo in CONFIGS:
    bad_opt = bad_naive = 0
    first_bad = None
    for f in range(-128, 128):
        for w in range(-128, 128):
            ref = original(f, w, fpf, fpw, fpo)
            if optimised(f, w, fpf, fpw, fpo) != ref:
                bad_opt += 1
                if first_bad is None:
                    first_bad = (f, w, ref, optimised(f, w, fpf, fpw, fpo))
            if naive(f, w, fpf, fpw, fpo) != ref:
                bad_naive += 1
    status = "MATCH" if bad_opt == 0 else "*** MISMATCH ***"
    print("%-18s shift=%d  optimised: %-16s (naive round-half-up would "
          "differ on %d)" % (name, fpf + fpw - fpo, status, bad_naive))
    if first_bad:
        print("     first mismatch: feat=%d w=%d ref=%d got=%d" % first_bad)
        all_ok = False

print()
print("RESULT:", "optimised IP is BIT-IDENTICAL to the original"
      if all_ok else "*** DIVERGENCE -- do not synthesise ***")
