#!/usr/bin/env python3
"""
probe_dpu_hang.py -- run ON THE BOARD. Decides WHY the DPU times out.

Two hypotheses, and they need completely different fixes:

  (H1) LOST INTERRUPT. The DPU finishes, but v++ cascades every CU's
       interrupt through axi_intc_0 into ONE PS line (pl_ps_irq0 = SPI 89 =
       GIC 121), while zocl gives each CU its own IRQ index (CU0 -> 121,
       CU1 -> 122). The DPU is CU index 1, so its done-interrupt lands on
       121 and nobody is waiting there.
       => fixable in the platform/device tree, no DPU rebuild.

  (H2) GENUINE STALL. The DPU never completes at all -- it cannot fetch
       instructions or data over its M_AXI ports.
       => needs a config/connectivity change and a re-link.

Discriminator: run one inference in a CHILD process while the parent
watches the DPU CU's own ap_ctrl register and the GIC counters.

  ap_done (bit 1) goes high  -> H1 (it finished; only the notify was lost)
  ap_ctrl stays busy/idle    -> H2 (it never ran)
  GIC 121 count increments   -> H1, and names the line it actually used

Read-only on the DPU: the parent only READS 0xa0010000. Starting the DPU
is left entirely to XRT in the child, exactly as VART would do it.
"""
import mmap
import os
import struct
import subprocess
import sys
import time

DPU_BASE = 0xA0010000
AP_CTRL = 0x00

CHILD = "/home/root/hybrid_pkg/_dpu_child.py"

CHILD_SRC = '''
import numpy as np, xir, vart
g = xir.Graph.deserialize("/home/root/lcam_hybrid.xmodel")
root = g.get_root_subgraph()
subs = [s for s in root.toposort_child_subgraph()
        if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]
r = vart.Runner.create_runner(subs[0], "run")
it, ot = r.get_input_tensors(), r.get_output_tensors()
ins  = [np.zeros(tuple(t.dims), dtype=np.int8) for t in it]
outs = [np.zeros(tuple(t.dims), dtype=np.int8) for t in ot]
print("child: submitting", flush=True)
r.wait(r.execute_async(ins, outs))
print("child: COMPLETED", flush=True)
'''


def gic_counts():
    out = {}
    with open("/proc/interrupts") as f:
        for line in f:
            if "zocl_irq_intc" in line:
                parts = line.split()
                idx = parts[0].rstrip(":")
                out[idx] = sum(int(x) for x in parts[1:5])
    return out


def main():
    with open(CHILD, "w") as f:
        f.write(CHILD_SRC)

    fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
    mm = mmap.mmap(fd, 0x1000, mmap.MAP_SHARED, mmap.PROT_READ, offset=DPU_BASE)

    def ap():
        return struct.unpack_from("<I", mm, AP_CTRL)[0]

    print("DPU ap_ctrl before   = 0x%x" % ap())
    before = gic_counts()
    print("GIC counts before    = %s" % before)

    env = dict(os.environ, XLNX_DPU_TIMEOUT="60000")
    p = subprocess.Popen([sys.executable, CHILD], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    print("\nwatching DPU ap_ctrl for 30 s ...")
    seen = {}
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 30.0:
        v = ap()
        if v not in seen:
            seen[v] = time.perf_counter() - t0
            print("  t=%6.2fs  ap_ctrl = 0x%x   (start=%d done=%d idle=%d ready=%d)"
                  % (seen[v], v, v & 1, (v >> 1) & 1, (v >> 2) & 1, (v >> 3) & 1))
        if p.poll() is not None:
            break
        time.sleep(0.01)

    after = gic_counts()
    print("\nGIC counts after     = %s" % after)
    for k in sorted(set(before) | set(after)):
        d = after.get(k, 0) - before.get(k, 0)
        print("  irq %-4s delta %d" % (k, d))

    try:
        p.wait(timeout=60)
    except Exception:
        p.kill()
    out = p.stdout.read().decode(errors="replace")
    print("\n--- child output ---")
    print(out[-2000:])

    print("\n--- VERDICT ---")
    done_seen = any((v >> 1) & 1 for v in seen)
    irq_fired = any(after.get(k, 0) - before.get(k, 0) > 0 for k in after)
    if done_seen or irq_fired:
        print("  H1: the DPU DID complete (ap_done seen=%s, irq fired=%s)."
              % (done_seen, irq_fired))
        print("      The compute is fine; the completion notification is")
        print("      being delivered on the wrong line. Fix in the platform")
        print("      IRQ wiring / device tree -- no DPU rebuild needed.")
    else:
        print("  H2: the DPU never completed -- ap_done never asserted and no")
        print("      interrupt fired. It is stalled on its own memory path,")
        print("      not merely missing a notification.")

    mm.close()
    os.close(fd)


if __name__ == "__main__":
    main()
