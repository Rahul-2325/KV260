# Custom HLS Accelerator Deployment for LCAM-YOLOX on KV260
### A plain-language explainer — what we're doing, what each tool is, and why

---

## The one-sentence version

We designed 9 custom hardware accelerator blocks (in place of Xilinx's off-the-shelf DPU)
to run our LCAM-YOLOX wildfire detection network, and the last several days of work
have been getting Linux on the board to correctly recognize and grant memory access to
those custom blocks — the actual neural network math has been done and verified since
day one; what's hard is the "plumbing" between hardware and software.

---

## 1. What "DPU" means and why we moved away from it

**DPU (Deep Learning Processing Unit)** is Xilinx's pre-built, general-purpose neural
network accelerator IP. You give it a compiled model and it runs the standard layers
(convolution, pooling, etc.) using a fixed instruction set — like a small, specialized
CPU built just for neural networks. It's fast and well-supported, but it's a black box:
you can't customize *how* it computes, only *what* model you feed it.

**Our approach:** instead of using DPU, we built 9 separate custom hardware blocks in
**HLS (High-Level Synthesis)** — a design flow where you write C++ describing a
computation, and a tool (Vitis HLS) automatically converts it into digital circuit logic
(RTL/Verilog) that runs on the FPGA fabric. Each block handles one type of operation our
network needs: `conv2d_engine` (convolutions), `pool_engine` (pooling),
`lcam_attention_gate` (our paper's attention mechanism), and so on.

**Why:** this is the actual research contribution — building purpose-specific hardware
for our specific network's operations, rather than relying on a general-purpose vendor
IP. It also means we can measure and report real performance/resource numbers for a
fully custom accelerator, which is a comparison point the base paper's DPU-only
results don't have.

---

## 2. Key terms, in the order you'll actually explain them

| Term | Plain definition |
|---|---|
| **PL** (Programmable Logic) | The actual FPGA fabric — reconfigurable digital circuits. This is where our 9 custom IPs physically live. |
| **PS** (Processing System) | The fixed ARM CPU cores on the same chip, running Linux. This is "the computer" that controls the PL. |
| **Bitstream (`.bit`)** | The compiled configuration file that programs the PL — tells the FPGA fabric exactly what circuit to become. Produced by Vivado. |
| **AXI** | The standard on-chip bus protocol connecting PS and PL. Two kinds we use: **AXI-Lite** (simple control registers — start/stop, parameters) and **AXI4** (high-speed bulk data transfer, e.g. reading/writing image tensors to memory). |
| **XSA** | A hardware description file exported from Vivado — a snapshot of "what IP exists, at what addresses, connected how." Feeds into later tools. |
| **Device tree / device tree overlay (`.dtbo`)** | A file that tells the Linux **kernel** what hardware exists on the board and where — CPU cores, memory, and (for us) our custom IPs' addresses. Without this, Linux has no idea our IPs exist and won't set up any drivers for them. |
| **xclbin** | A metadata container (built by Xilinx's `xclbinutil`) describing two things: (1) `MEM_TOPOLOGY` — which memory banks/DDR ports are in use, and (2) `IP_LAYOUT` — where each accelerator's control registers live. This is what tells the *runtime* (not the kernel, the user-space memory manager) how to allocate buffers that both the ARM CPU and the FPGA can both see. |
| **zocl** | The specific Linux kernel driver for Xilinx Zynq FPGAs. Its job: create a `/dev/dri/renderDxxx` device file, and through it, let user programs (like our Python scripts) allocate **physically contiguous memory** that the PL's hardware can directly read/write — this is essential because the FPGA can't use normal virtual memory the way a CPU program can. |
| **xmutil / dfx-mgr** | Xilinx's official Kria-board utility + background service for loading/unloading a PL configuration ("app") at runtime, replacing an older, more error-prone method of just raw-flashing a bitstream. |
| **CMA (Contiguous Memory Allocator)** | A Linux kernel feature for reserving blocks of physically contiguous RAM — needed because DMA-capable hardware (like our IPs) can't work with the scattered memory pages normal software uses. |

---

## 3. Why the deployment took several days — the actual technical story

This is the part worth explaining carefully, because it's the genuine engineering
narrative, not just "it didn't work":

**Attempt 1 — raw bitstream load.** We used a low-level tool (`fpgautil`) to program
the bitstream directly onto the FPGA. The IPs powered on and responded to control
signals correctly — but every attempt to actually read/write data (tensors, weights)
silently failed. No error anywhere. This is because `fpgautil` **bypasses** the whole
device-tree/zocl/xclbin chain above — Linux never even knew our IPs existed as
DMA-capable devices, so it never set up a way for them to safely access memory.

**Diagnosis (this took real systematic work):** we ran a sequence of isolated hardware
tests — checking that control registers responded, that memory buffers were valid,
that the hardware genuinely executed (not just returning cached "done" signals) — and
progressively ruled out cache coherency, addressing bugs, and interconnect wiring, one
at a time, until only one explanation remained: **the proper OS-level loading path was
never used.**

**Attempt 2 — the correct path.** We built the full chain shown in the diagram above:
exported the hardware description (XSA), generated a device tree overlay describing
our IPs to Linux, built an xclbin with the memory/IP metadata the runtime needs, and
packaged everything through Xilinx's official `xmutil`/`dfx-mgr` loading mechanism —
the same system used by Xilinx's own pre-built example accelerators. This is currently
where we are: the loading pipeline is fully built and each individual piece has been
verified working; we're in the final step of confirming the memory-access driver
(`zocl`) initializes correctly for our design specifically.

---

## 4. What to say if asked "so does it work yet?"

Be precise: **the hardware design is fully verified in simulation** (every one of the 9
custom IPs individually passed C-simulation with known-correct outputs), **the
bitstream synthesizes and implements cleanly on real silicon**, and **we've built the
complete, correct deployment pipeline that Xilinx's own tools use for custom
accelerators.** What remains is confirming the final memory-access handshake on real
hardware — we are actively debugging the last link in a chain we've now fully
constructed, not searching blind.

---

## 5. If your professor asks "why not just use DPU normally?"

Fair question — the honest answer: DPU already works and is fast (the base paper
reports 195 FPS on this exact board). The reason to build custom IPs instead is
**research novelty and control** — DPU can't be modified to implement a paper's novel
architecture (LCAM attention) as native hardware, it can only run it as a slower
software fallback on the ARM CPU alongside DPU. Building custom accelerators lets us
report real hardware numbers for the *whole* network including the novel parts, and
contributes a documented method (xmodel weight/topology extraction + full custom HLS
deployment) that goes beyond what the base paper covers.
