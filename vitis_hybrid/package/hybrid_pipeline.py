#!/usr/bin/env python3
"""
hybrid_pipeline.py -- run ON THE BOARD, with kv260-hybrid2 loaded.

The hand-written replacement for GraphRunner: walks the 24 subgraphs of
lcam_hybrid.xmodel in topological order, runs the 8 DPU subgraphs on the
DPU, and executes the 15 CPU subgraphs itself -- sending the four LCAM
attention gates to the custom CU at 0xa0020000 instead of the ARM core.

WHY BY HAND: GraphRunner dispatches "DPU subgraph -> DPU, everything else
-> CPU" internally and exposes no hook to redirect one op to a custom CU.
It is also slow at the ops it keeps: measured, VART spends 91 ms
transposing a 58,800-element tensor that numpy does in 1.3 ms.

Modes:
  --gates ip      four gates on the custom CU        (the point of all this)
  --gates numpy   four gates in numpy                (control / correctness)
Both reproduce the same result; the difference is time.

Correctness is checked against the GraphRunner output saved earlier in
/home/root/e2e_outputs.npz.

REQUIRES  XRT_INI_PATH=/home/root/xrt.ini  ([Runtime] ert_polling=true) --
without it every DPU call times out after 10 s (PROJECT_HISTORY 33.2).
"""
import argparse
import mmap
import os
import struct
import sys
import time

import numpy as np
import vart
import xir

sys.path.insert(0, "/home/root")
from hw_ip_driver import ZoclBuffer, DRM_PATH_DEFAULT

XMODEL = "/home/root/lcam_hybrid.xmodel"

# ---------------------------------------------------------------- LCAM CU
CU_BASE = 0xA0020000
R_AP, R_FIN, R_WIN, R_FOUT = 0x00, 0x10, 0x1C, 0x28
R_H, R_W, R_C, R_FPF, R_FPW, R_FPO = 0x34, 0x3C, 0x44, 0x4C, 0x54, 0x5C
AP_START, AP_DONE, AP_IDLE, AP_CONTINUE = 0x1, 0x2, 0x4, 0x10


class LcamCU(object):
    """
    AXI-Lite driver for the v++-linked LCAM kernel.

    NOTE ap_ctrl_chain: ap_done is STICKY and must be acknowledged with
    ap_continue. The 9-IP driver's poll_ap_done() assumes ap_ctrl_hs and
    will silently no-op every run after the first (PROJECT_HISTORY 32.3).
    """

    def __init__(self):
        self.fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, 0x1000, mmap.MAP_SHARED,
                            mmap.PROT_READ | mmap.PROT_WRITE, offset=CU_BASE)
        # 2 MB is ample: the largest gate tensor is 160*160*64 = 1,638,400 B.
        # The SIZE MATTERS BEYOND CAPACITY -- see the sync note in gate().
        NBUF = 2 << 20
        self.bi = ZoclBuffer(DRM_PATH_DEFAULT, NBUF)
        self.bw = ZoclBuffer(DRM_PATH_DEFAULT, NBUF)
        self.bo = ZoclBuffer(DRM_PATH_DEFAULT, NBUF)
        for b in (self.bi, self.bw, self.bo):
            if b.phys_addr >= 0x80000000:
                raise MemoryError("buffer in HIGH DDR, PL cannot reach it")

        # Direct int8 views over each buffer's mapping. ZoclBuffer's
        # write_array() goes ndarray -> tobytes() -> mmap slice, i.e. TWO
        # full copies, and read_array() does bytes() then frombuffer().copy(),
        # two more. Assigning through a view is one copy each way.
        self.vi = np.frombuffer(self.bi._mm, dtype=np.int8)
        self.vw = np.frombuffer(self.bw._mm, dtype=np.int8)
        self.vo = np.frombuffer(self.bo._mm, dtype=np.int8)

        # NOTE: a cached second mapping of the output buffer (/dev/mem without
        # O_SYNC) reads 18x faster -- 2432 vs 137 MB/s -- and was tried here.
        # It is NOT COHERENT and produced wrong results; see the long note in
        # gate(). Deliberately not created. ??37.3.
        self.t_in = self.t_run = self.t_out = self.t_sync = 0.0

    def _w32(self, off, v):
        struct.pack_into("<I", self.mm, off, v & 0xFFFFFFFF)

    def _r32(self, off):
        return struct.unpack_from("<I", self.mm, off)[0]

    def _w64(self, off, v):
        self._w32(off, v & 0xFFFFFFFF)
        self._w32(off + 4, (v >> 32) & 0xFFFFFFFF)

    def gate(self, feat, gate, fpf, fpw, fpo):
        """feat [1,H,W,C] int8, gate [1,H,W,1] int8 -> [1,H,W,C] int8."""
        _, H, W, C = feat.shape
        t0 = time.perf_counter()
        while not (self._r32(R_AP) & AP_IDLE):
            if time.perf_counter() - t0 > 5:
                raise RuntimeError("CU not idle, ap_ctrl=0x%x" % self._r32(R_AP))

        nf, nw = H * W * C, H * W
        tA = time.perf_counter()
        # SYNC ONLY THE BYTES ACTUALLY USED. sync_to_device()/sync_from_device()
        # default to the FULL buffer, so 8 MB buffers meant 4 gates x 2 syncs
        # x 8 MB = 64 MB of cache maintenance per frame for ~6 MB of real
        # data. That, not the DMA, was most of the 28 ms of gate overhead.
        self.vi[:nf] = feat.reshape(-1)
        self.bi.sync_to_device(size=nf)
        self.vw[:nw] = gate.reshape(-1)
        self.bw.sync_to_device(size=nw)

        tB = time.perf_counter()

        self._w64(R_FIN, self.bi.phys_addr)
        self._w64(R_WIN, self.bw.phys_addr)
        self._w64(R_FOUT, self.bo.phys_addr)
        for off, val in ((R_H, H), (R_W, W), (R_C, C),
                         (R_FPF, fpf), (R_FPW, fpw), (R_FPO, fpo)):
            self._w32(off, val)

        self._w32(R_AP, AP_START)
        t0 = time.perf_counter()
        while not (self._r32(R_AP) & AP_DONE):
            if time.perf_counter() - t0 > 5:
                raise RuntimeError("CU timeout")
        self._w32(R_AP, AP_CONTINUE)          # mandatory for ap_ctrl_chain
        tC = time.perf_counter()

        self.bo.sync_from_device(size=nf)

        tC2 = time.perf_counter()
        # One copy out of the mapping. The copy is required: this buffer is
        # reused by the next gate, and the value must survive until the
        # following DPU subgraph consumes it.
        # READ THROUGH THE WRITE-COMBINING VIEW, not self.vo_cached.
        #
        # The cached mapping reads 18x faster (2432 vs 137 MB/s) and took the
        # frame to 54.83 ms / 18.24 FPS -- but it returned WRONG DATA: 10414
        # of 58800 output elements differed from the numpy control. It is not
        # coherent. sync_from_device() appears to be a no-op for these CMA
        # buffers (memory allocated coherent needs no maintenance, so the
        # kernel skips the cache op), leaving stale lines in our mapping.
        #
        # verify_cached_read.py passed 12/12 and was WRONG to: it poisoned the
        # buffer through the write-combining view each iteration and cycled
        # sizes up to 1.6 MB, larger than the 1 MB L2, so cache pressure
        # evicted the stale lines for it. It never reproduced this access
        # pattern. Do not re-enable on the strength of that test.
        res = self.vo[:nf].reshape(1, H, W, C).copy()

        tD = time.perf_counter()

        # phase accounting: where the gate wall-time actually goes

        self.t_in += tB - tA

        self.t_run += tC - tB

        self.t_sync += tC2 - tC

        self.t_out += tD - tC2

        return res

    def close(self):
        self.mm.close()
        os.close(self.fd)
        for b in (self.bi, self.bw, self.bo):
            b.close()


# ------------------------------------------------------------- numpy ops
def dpu_round(x):
    """DPU_ROUND = round half AWAY FROM ZERO (numpy rounds half to even)."""
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))


def f2fix(x, fp):
    return np.clip(dpu_round(x * (2.0 ** fp)), -128, 127).astype(np.int8)


def fix2f(x, fp):
    return x.astype(np.float32) * np.float32(2.0 ** -fp)


def run_op(op, vals):
    t = op.get_type()
    ins = [vals[i.name] for i in op.get_input_tensors()]
    ot = op.get_output_tensor()

    def fp_of(o, tensor):
        if o.has_attr("fix_point"):
            return o.get_attr("fix_point")
        return tensor.get_attr("fix_point")

    if t == "fix2float":
        return fix2f(ins[0], fp_of(op, op.get_input_tensors()[0]))
    if t == "float2fix":
        return f2fix(ins[0], fp_of(op, ot))
    if t == "mul":
        return ins[0] * ins[1]
    if t == "sigmoid":
        return (1.0 / (1.0 + np.exp(-ins[0]))).astype(np.float32)
    if t == "transpose":
        return np.ascontiguousarray(np.transpose(ins[0], op.get_attr("order")))
    if t in ("reshape", "reshape-fix"):
        return np.ascontiguousarray(ins[0]).reshape(ot.dims)
    if t in ("concat", "concat-fix"):
        return np.concatenate(ins, axis=op.get_attr("axis"))
    if t == "fix":
        # simulated quantisation: snap to the fixed-point grid, stay float
        fp = fp_of(op, ot)
        return (np.clip(dpu_round(ins[0] * (2.0 ** fp)), -128, 127)
                * np.float32(2.0 ** -fp)).astype(np.float32)
    if t in ("download", "upload", "identity"):
        return ins[0]
    raise NotImplementedError("op type %r not implemented" % t)


def run_cpu_subgraph(sg, vals):
    """Execute a CPU subgraph's ops, resolving order by data dependency."""
    pending = list(sg.get_ops())
    guard = 0
    while pending:
        progressed = False
        for op in list(pending):
            if all(i.name in vals for i in op.get_input_tensors()):
                vals[op.get_output_tensor().name] = run_op(op, vals)
                pending.remove(op)
                progressed = True
        guard += 1
        if not progressed or guard > 200:
            raise RuntimeError("cannot order ops in subgraph %s" % sg.get_name())


def gate_params(sg):
    """Pull (fp_feat, fp_weight, fp_out) and tensor names out of a gate subgraph."""
    feat_t = wt_t = None
    fpf = fpw = fpo = None
    for op in sg.get_ops():
        if op.get_type() == "fix2float":
            src = op.get_input_tensors()[0]
            fp = op.get_attr("fix_point") if op.has_attr("fix_point") \
                else src.get_attr("fix_point")
            if src.dims[-1] == 1:
                wt_t, fpw = src.name, fp
            else:
                feat_t, fpf = src.name, fp
        elif op.get_type() == "float2fix":
            ot = op.get_output_tensor()
            fpo = op.get_attr("fix_point") if op.has_attr("fix_point") \
                else ot.get_attr("fix_point")
    return feat_t, wt_t, fpf, fpw, fpo


def sg_out(sg):
    """xir returns subgraph output tensors as a SET, which has no order and
    is not subscriptable. Sort by name so the ordering is deterministic."""
    return sorted(sg.get_output_tensors(), key=lambda t: t.name)


def build_int8_path(sg):
    """
    Strip a pointless float round-trip off a shape-only subgraph.

    Three subgraphs are  fix2float(fp) -> transpose -> float2fix(fp)  with the
    SAME fix_point on both ends, and one more is fix2float(fp) -> transpose ->
    fix(fp). Dequantising by 2^-fp and requantising by 2^fp is exactly the
    identity for int8 input: both are exact powers of two, float32 represents
    every int8 exactly, and nothing can leave [-128,127]. So the whole
    subgraph is just a transpose -- of int8, not of float32.

    That matters: these were 5.58 + 3.97 + 1.54 + 1.19 = 12.3 ms/frame,
    moving 4x the bytes through float32 and building several temporaries per
    op. Done on int8 directly they are a single ascontiguousarray.

    Returns (in_name, out_name, fn) or None.
    """
    ops = list(sg.get_ops())
    types = sorted(op.get_type() for op in ops)
    if types not in (["fix2float", "float2fix", "transpose"],
                     ["fix", "fix2float", "transpose"]):
        return None

    f2f = next(o for o in ops if o.get_type() == "fix2float")
    shape_op = next(o for o in ops if o.get_type() == "transpose")
    tail = next(o for o in ops if o.get_type() in ("float2fix", "fix"))

    def fp_of(o, tensor):
        return o.get_attr("fix_point") if o.has_attr("fix_point") \
            else tensor.get_attr("fix_point")

    fp_in = fp_of(f2f, f2f.get_input_tensors()[0])
    fp_out = fp_of(tail, tail.get_output_tensor())
    if fp_in != fp_out:
        return None                     # scales differ, the round-trip is real

    produced = set(o.get_output_tensor().name for o in ops)
    ins = [t.name for o in ops for t in o.get_input_tensors()
           if t.name not in produced]
    outs = sg_out(sg)
    if len(set(ins)) != 1 or len(outs) != 1:
        return None

    order = shape_op.get_attr("order")
    emits_float = tail.get_type() == "fix"
    scale = np.float32(2.0 ** -fp_in)

    def fn(x):
        t = np.ascontiguousarray(np.transpose(x, order))
        return t.astype(np.float32) * scale if emits_float else t

    return list(set(ins))[0], outs[0].name, fn


ELEMENTWISE = ("fix2float", "float2fix", "sigmoid", "relu", "fix")


def build_lut(sg):
    """
    Collapse a whole CPU subgraph into a 256-entry int8 -> int8 lookup table.

    Six subgraphs are fix2float -> sigmoid -> float2fix over a single int8
    tensor. int8 has only 256 possible values and every op is elementwise, so
    the entire chain IS a 256-entry table. Measured, those six cost ~7.3 ms
    per frame in numpy -- subgraph [10] alone spends 1.17 ms on a 400-element
    tensor, i.e. almost entirely per-call overhead. This is precisely why
    VART beats numpy on these ops (??34): it uses a LUT too.

    Exact by construction: the table is filled by running the very same numpy
    ops over all 256 possible inputs.

    Returns (in_name, out_name, table) or None if the subgraph does not fit.
    """
    ops = list(sg.get_ops())
    if not ops or any(op.get_type() not in ELEMENTWISE for op in ops):
        return None

    produced = set(op.get_output_tensor().name for op in ops)
    ins = []
    for op in ops:
        for t in op.get_input_tensors():
            if t.name not in produced and t.name not in ins:
                ins.append(t.name)
    outs = sg_out(sg)
    if len(ins) != 1 or len(outs) != 1:
        return None

    # probe[i] is the int8 whose uint8 reinterpretation is i, so the table can
    # be indexed directly with x.view(np.uint8) at run time.
    probe = np.arange(256, dtype=np.uint8).view(np.int8)
    v = {ins[0]: probe}
    try:
        run_cpu_subgraph(sg, v)
    except Exception:
        return None
    res = v[outs[0].name]
    if res.dtype != np.int8 or res.shape != probe.shape:
        return None
    return ins[0], outs[0].name, res


def is_gate(sg):
    types = sorted(op.get_type() for op in sg.get_ops())
    return types == ["fix2float", "fix2float", "float2fix", "mul"]


# ------------------------------------------------------------------ main
def letterbox(img, w=640, h=640):
    """Returns the canvas AND the scale/pad, which postprocessing needs to
    map boxes back onto the original image."""
    import cv2
    oh, ow = img.shape[:2]
    s = min(w / ow, h / oh)
    nw, nh = int(ow * s), int(oh * s)
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((h, w, 3), 114, dtype=np.uint8)
    px, py = (w - nw) // 2, (h - nh) // 2
    canvas[py:py + nh, px:px + nw] = r
    return canvas, s, px, py, ow, oh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", choices=("ip", "numpy"), default="ip")
    ap.add_argument("--image", default="/home/root/WEB09971.jpg")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--round", choices=("ref", "dpu"), default="ref",
                    help="input quantisation rounding; ref = np.round, matches e2e_outputs.npz")
    ap.add_argument("--save-npz", dest="save_npz", default=None,
                    help="write out_0 + letterbox meta for postprocess_detections.py")
    ap.add_argument("--nolut", action="store_true",
                    help="disable the int8 LUT collapse, for A/B")
    ap.add_argument("--save", default=None,
                    help="write the final output tensor here, to A/B the gate modes")
    ap.add_argument("--out", default=None,
                    help="ONE-COMMAND DEMO: decode + NMS + draw + save an "
                         "annotated image here. Runs postprocess_detections."
                         "run_postprocess() right after the timing loop -- no "
                         "separate command, no .npz round-trip. The FPS "
                         "printed above still measures ONLY the hardware "
                         "pipeline; postprocessing's own cost is measured "
                         "and printed separately, never folded in.")
    ap.add_argument("--conf", type=float, default=0.30, help="used with --out")
    ap.add_argument("--nms", type=float, default=0.45, help="used with --out")
    ap.add_argument("--classes", default="smoke,fire", help="used with --out")
    ap.add_argument("--ref", default="/home/root/e2e_outputs.npz")
    args = ap.parse_args()

    import cv2
    g = xir.Graph.deserialize(XMODEL)
    subs = g.get_root_subgraph().toposort_child_subgraph()

    runners, kinds = {}, {}
    for i, s in enumerate(subs):
        dev = s.get_attr("device").upper() if s.has_attr("device") else "USER"
        kinds[i] = dev
        if dev == "DPU":
            runners[i] = vart.Runner.create_runner(s, "run")
    print("subgraphs: %d  (DPU %d, CPU %d)"
          % (len(subs), sum(1 for v in kinds.values() if v == "DPU"),
             sum(1 for v in kinds.values() if v == "CPU")))

    # Precompute int8->int8 lookup tables for the elementwise CPU subgraphs.


    luts = {}


    if not args.nolut:


        for i, s in enumerate(subs):


            if kinds[i] == "CPU" and not is_gate(s):


                r = build_lut(s)

                if r is not None:

                    luts[i] = ("lut",) + r

                else:

                    r = build_int8_path(s)

                    if r is not None:
                        luts[i] = ("int8",) + r


        print("optimised CPU subgraphs: %d of %d (LUT or int8 fast path)"


              % (len(luts), sum(1 for k, v in kinds.items() if v == "CPU")))



    cu = LcamCU() if args.gates == "ip" else None
    if cu:
        print("LCAM CU @0x%08x  buffers in low DDR" % CU_BASE)

    img = cv2.imread(args.image)
    canvas, lb_scale, lb_px, lb_py, orig_w, orig_h = letterbox(img)

    in_name = sg_out(subs[0])[0].name
    in_fp = sg_out(subs[0])[0].get_attr("fix_point")
    # INPUT ROUNDING MUST MATCH THE REFERENCE, or the comparison is
    # meaningless. fix_point is -1 here, so quantising is pixel/2 and every
    # ODD pixel value lands exactly on .5 -- precisely where np.round
    # (banker's, ties-to-even) and DPU_ROUND (ties-away-from-zero) disagree.
    # e2e_graphrunner.py, which produced e2e_outputs.npz, used np.round, so
    # --round ref reproduces it bit for bit. --round dpu is what the DPU
    # spec actually says and is the correct choice for deployment; it just
    # cannot be compared against this particular reference file.
    rnd = np.round if args.round == "ref" else dpu_round
    quant = np.clip(rnd(canvas.astype(np.float32) * (2.0 ** in_fp)),
                    -128, 127).astype(np.int8).reshape(1, 640, 640, 3)

    times, gate_ms, dpu_ms, cpu_ms = [], 0.0, 0.0, 0.0

    sg_ms = {}
    out_final = None

    for run in range(args.runs + 1):          # first is warm-up
        vals = {in_name: quant}
        tg = td = tc = 0.0
        t_start = time.perf_counter()

        for i, s in enumerate(subs):
            if kinds[i] == "USER":
                continue
            if kinds[i] == "DPU":
                r = runners[i]
                its, ots = r.get_input_tensors(), r.get_output_tensors()
                ins = [np.ascontiguousarray(vals[t.name]) for t in its]
                outs = [np.zeros(tuple(t.dims), dtype=np.int8) for t in ots]
                t0 = time.perf_counter()
                r.wait(r.execute_async(ins, outs))
                td += time.perf_counter() - t0
                for t, o in zip(ots, outs):
                    vals[t.name] = o
            else:
                if cu is not None and is_gate(s):
                    ft, wt, fpf, fpw, fpo = gate_params(s)
                    t0 = time.perf_counter()
                    res = cu.gate(vals[ft], vals[wt], fpf, fpw, fpo)
                    tg += time.perf_counter() - t0
                    vals[sg_out(s)[0].name] = res
                else:
                    t0 = time.perf_counter()
                    if i in luts:
                        kind, src, dst, f = luts[i]
                        vals[dst] = (f[vals[src].view(np.uint8)]
                                     if kind == "lut" else f(vals[src]))
                    else:
                        run_cpu_subgraph(s, vals)
                    dt = time.perf_counter() - t0
                    tc += dt
                    sg_ms[i] = sg_ms.get(i, 0.0) + dt

        total = time.perf_counter() - t_start
        out_final = vals[sg_out(subs[-1])[0].name]
        if run == 0:
            continue
        times.append(total * 1e3)
        gate_ms += tg * 1e3
        dpu_ms += td * 1e3
        cpu_ms += tc * 1e3

    n = len(times)
    print()
    print("=" * 62)
    print("HAND-WRITTEN PIPELINE   gates = %s" % args.gates)
    print("=" * 62)
    print("  DPU  (8 subgraphs)     : %7.2f ms" % (dpu_ms / n))
    print("  gates (4)              : %7.2f ms" % (gate_ms / n))
    print("  other CPU subgraphs    : %7.2f ms" % (cpu_ms / n))
    if cu is not None:
        print("     gate breakdown: write+sync %.2f | CU run %.2f | sync_from %.2f | read %.2f ms"
              % (cu.t_in * 1e3 / (n + 1), cu.t_run * 1e3 / (n + 1), cu.t_sync * 1e3 / (n + 1), cu.t_out * 1e3 / (n + 1)))
    print("  -----------------------------------")
    print("  mean total             : %7.2f ms   -> %.2f FPS"
          % (np.mean(times), 1000.0 / np.mean(times)))
    print("  min / max              : %7.2f / %.2f ms" % (min(times), max(times)))
    print()
    if sg_ms:

        print()

        print("  [check] sum(sg_ms)/n = %.3f ms   vs cpu_ms/n = %.3f ms   n=%d  entries=%d"

              % (sum(sg_ms.values()) * 1e3 / (n + 1), cpu_ms / n, n, len(sg_ms)))

        print("  per-CPU-subgraph (ms/frame), worst first:")

        for idx, v in sorted(sg_ms.items(), key=lambda kv: -kv[1])[:8]:

            types = sorted(set(op.get_type() for op in subs[idx].get_ops()))

            shp = list(sg_out(subs[idx])[0].dims)

            print("    [%02d] %-8.3f %-22s %s" % (idx, v * 1e3 / (n + 1), str(shp), ",".join(types)))

        print()

    print("  GraphRunner baseline   :  426.51 ms   -> 2.34 FPS")
    print("  speedup                : %7.2fx" % (426.51 / np.mean(times)))

    if os.path.exists(args.ref):
        ref = np.load(args.ref)
        key = [k for k in ref.files if k.startswith("out")][0]
        r = ref[key].astype(np.float32).reshape(-1)
        o = np.asarray(out_final, dtype=np.float32).reshape(-1)
        if r.shape == o.shape:
            d = np.abs(r - o)
            print()
            print("  vs GraphRunner output: max|diff| = %.6g   mean = %.6g"
                  % (d.max(), d.mean()))
            print("  %s" % ("MATCH" if d.max() < 1e-3 else "*** DIFFERS ***"))
        else:
            print("\n  ref shape %s != ours %s" % (r.shape, o.shape))

    if args.save_npz:
        # Same keys postprocess_detections.py expects from e2e_graphrunner.py,
        # so the existing decoder/NMS works on this pipeline's output unchanged.
        np.savez(args.save_npz,
                 out_0=np.asarray(out_final, dtype=np.float32),
                 scale=np.float32(lb_scale),
                 pad=np.array([lb_px, lb_py], dtype=np.int32),
                 orig=np.array([orig_w, orig_h], dtype=np.int32),
                 latency_ms=np.array(times, dtype=np.float32))
        print("\n  saved detections tensor + meta -> %s" % args.save_npz)



    if args.save:


        np.save(args.save, np.asarray(out_final, dtype=np.float32))


        print("\n  saved final output -> %s" % args.save)

    if args.out:
        # ONE COMMAND, image in -> boxes out. Deliberately called AFTER the
        # timing loop above and AFTER `times`/FPS have been computed and
        # printed: this step's own cost (measured below, not hidden) never
        # touches the hardware throughput number.
        print()
        print("  postprocessing (decode + NMS + draw)...")
        import postprocess_detections as pp
        pp.run_postprocess(out_final, lb_scale, lb_px, lb_py, orig_w, orig_h,
                           args.image, args.out, args.conf, args.nms,
                           args.classes, verbose=True)

    if cu:
        cu.close()


if __name__ == "__main__":
    main()












