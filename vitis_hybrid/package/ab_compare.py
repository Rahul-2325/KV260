#!/usr/bin/env python3
"""
ab_compare.py -- compare the pipeline's output with the gates on the custom
CU against the same pipeline with the gates in numpy.

THIS is the correctness claim that matters for the IP: everything else in
the two runs is identical code, so any difference is the IP's doing.

Comparing against GraphRunner instead is NOT a clean test -- VART computes
sigmoid with a lookup table over the 256 possible int8 inputs (that is why
it manages 12,800 elements in 0.05 ms, far too fast for expf), whereas this
pipeline evaluates sigmoid exactly. That difference is unrelated to the IP.
"""
import sys

import numpy as np

a = np.load(sys.argv[1] if len(sys.argv) > 1 else "/tmp/out_ip.npy")
b = np.load(sys.argv[2] if len(sys.argv) > 2 else "/tmp/out_np.npy")

d = np.abs(a - b)
print()
print("IP-gates vs numpy-gates   shape %s" % (a.shape,))
print("  max|diff|        = %g" % d.max())
print("  mean|diff|       = %g" % d.mean())
print("  mismatched elems = %d of %d" % (int((d > 0).sum()), d.size))
print("  -> %s" % ("IDENTICAL - the IP reproduces the reference exactly"
                   if d.max() == 0 else "DIFFERS"))
