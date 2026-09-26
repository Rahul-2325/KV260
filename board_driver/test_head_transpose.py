#!/usr/bin/env python3
"""
test_head_transpose.py
Single-IP hardware smoke test BEFORE trusting the full 153-layer run.

head_transpose is the safest first test: pure data movement (no math),
so any register-offset mistake shows up immediately and unambiguously
as a wrong permutation pattern, not a subtly-wrong number.

Run: sudo python3 test_head_transpose.py
"""

import numpy as np
import sys
from hw_ip_driver import ZoclBuffer, DRM_PATH_DEFAULT
from ip_wrappers import HeadTranspose


def golden_transpose(x, H, W, C):
    # [1,C,H,W] -> [1,H,W,C]  (matches the HLS source exactly)
    out = np.zeros((H, W, C), dtype=np.int8)
    for c in range(C):
        for h in range(H):
            for w in range(W):
                out[h, w, c] = x[c, h, w]
    return out


def main():
    # Small size for a fast, easy-to-debug test
    H, W, C = 4, 4, 3
    total = H * W * C

    # Distinct values 0..total-1 so any wrong index is immediately visible
    src = np.arange(total, dtype=np.int8).reshape(C, H, W)
    expected = golden_transpose(src, H, W, C)

    print(f"Testing head_transpose: [1,{C},{H},{W}] -> [1,{H},{W},{C}]")
    print(f"Input (C,H,W) flat: {src.flatten().tolist()}")

    in_buf  = ZoclBuffer(DRM_PATH_DEFAULT, total)
    out_buf = ZoclBuffer(DRM_PATH_DEFAULT, total)

    in_buf.write_array(src)

    ip = HeadTranspose()
    try:
        elapsed = ip.run(in_buf.phys_addr, out_buf.phys_addr, H, W, C)
        print(f"HW call completed in {elapsed*1000:.3f} ms")
    except TimeoutError as e:
        print(f"TIMEOUT: {e}")
        print("This means the register offsets are wrong for this IP -- "
              "ap_start was written but ap_done never came back.")
        sys.exit(1)
    finally:
        ip.close()

    result = out_buf.read_array((H, W, C), np.int8)

    print(f"\nExpected (H,W,C) flat: {expected.flatten().tolist()}")
    print(f"HW result (H,W,C) flat: {result.flatten().tolist()}")

    if np.array_equal(result, expected):
        print("\nPASS -- head_transpose register offsets and hardware "
              "both confirmed correct.")
    else:
        mismatches = np.sum(result != expected)
        print(f"\nFAIL -- {mismatches}/{total} elements wrong.")
        print("If ALL elements are 0 or garbage: register offsets are "
              "likely wrong (writes going to reserved/wrong registers).")
        print("If SOME elements match: possible partial addressing or "
              "AXI master routing issue in the block design.")
        sys.exit(1)

    in_buf.close()
    out_buf.close()


if __name__ == "__main__":
    if __import__("os").geteuid() != 0:
        raise SystemExit("Run as root: sudo python3 test_head_transpose.py")
    main()
