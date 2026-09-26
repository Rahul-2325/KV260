# LCAM-YOLOX Custom IP Project — 3-Day Engineering Summary
### Covering: HLS IP development → Vivado integration → board deployment → hardware debugging
### Status as of end of Day 3: BLOCKED on PS firewall issue, root cause identified, fix path chosen

---

## 1. THE GOAL

Replace parts of the LCAM-YOLOX inference pipeline (currently split between Xilinx's DPU and ARM CPU software) with custom HLS hardware accelerators on the KV260 board, to eliminate the ARM CPU bottleneck.

Two parallel approaches were explored:
- **Approach A**: Keep DPU for backbone convolutions, add custom IPs for the 15 CPU-bound subgraphs (attention gates, sigmoid, reshape, concat)
- **Approach B**: Replace DPU entirely — 9 custom IPs handle everything (all 111+ conv layers via a parameterized engine, plus pooling/eltwise/upsample/attention/head ops), driven by an ARM Python orchestrator reading a JSON layer plan extracted from the xmodel

---

## 2. THE 9 CUSTOM HLS IPs (all built, all C-sim PASSED)

| IP | Purpose | Covers |
|---|---|---|
| `conv2d_engine` | Parameterized INT8 convolution (1x1, 3x3, 7x7, strided) | All 111 conv2d-fix layers |
| `depthwise_engine` | Depthwise convolution | 1 depthwise-fix layer |
| `eltwise_add` | Residual/skip connections | eltwise-fix ops |
| `pool_engine` | Global average pooling | LCAM channel attention pooling |
| `upsample_engine` | Nearest-neighbor 2x upsample | FPN/PAN neck |
| `lcam_attention_gate` | Attention gate apply (mul + requant) | CPU subgraphs 02,04,06,08 |
| `head_sigmoid` | Sigmoid activation | CPU subgraphs 10-13,16,19 |
| `head_transpose` | NCHW→NHWC reshape | CPU subgraphs 15,18,21 |
| `head_concat_reshape` | Final detection concat | CPU subgraphs 22,23 |

All built in **Vitis HLS 2022.2**, part `xck26-sfvc784-2LV-c`, clock `10ns`.
Resource total: ~191 DSP (15%), ~74K LUT (62%) — fits comfortably alongside or instead of DPU.

**Vivado IP repo location:** `C:\Xilinx\projects\custom_ip_repo\` (also individual project folders at `C:\Xilinx\projects\<ip_name>\`)

---

## 3. DATA EXTRACTED FROM THE XMODEL (for Approach B)

Extracted inside the Vitis-AI 3.0 Docker container from `lcam_v5.xmodel`:

| File | Location | Contents |
|---|---|---|
| `layer_plan.json` | `~/wildfire_project/weights/` | 153 layers in execution order, each with op_type, shapes, stride/pad/kernel, fix_points |
| `weight_map.json` | `~/wildfire_project/weights/` | Maps each layer to its weight `.npy` file(s) |
| `weights/data/*.npy` | `~/wildfire_project/weights/data/` | 211 extracted INT8 weight/bias tensors, 8.6 MB total |

**Extraction method note:** xmodel `const-fix` op weights aren't in `get_attr('data')` (returns empty bytes) — they're embedded in the xmodel binary itself at a `ddr_addr` offset. Extraction script reads the raw xmodel file and locates the weight section by offset (see Day 2 notes — the `weight_section_start` heuristic worked, recovering 217/222 weight tensors, 211 unique files saved).

---

## 4. TIMELINE — WHAT HAPPENED EACH DAY

### Day 1 — Approach A build (DPU + 1 custom IP)
- Built `lcam_attention_gate` HLS IP, C-sim passed
- Integrated into existing KV260 Vivado project (DPU + decode_0 + nms_top_0 already present)
- Hit major routing congestion (16 AXI masters through 1 smartconnect → PL too congested to route)
- **Fixed** by removing `head_concat_reshape` from that attempt and reducing to 8 AXI ports total
- **Synthesis + implementation succeeded** — bitstream `top_wrapper.bit` (Jul 24, DPU + lcam_attention_gate only)
- Converted to `.bit.bin`, deployed to board via `fpgautil`, confirmed loads successfully
- **Never actually exercised in a real inference run** — ran `vitis_ai_library.GraphRunner` afterward but `/etc/vart.conf` was still pointing at the stock `kv260-benchmark-b4096.xclbin`, so `lcam_attention_gate` was never in the execution path. Full-pipeline inference (2.3 FPS, real detections) used the **stock DPU bitstream**, not our custom one.

### Day 2 — Pivot to Approach B (full custom, no DPU)
- Extracted `layer_plan.json` + weights from xmodel (documented above)
- Built all remaining 8 HLS IPs (conv2d_engine, depthwise_engine, eltwise_add, pool_engine, upsample_engine, head_sigmoid, head_transpose, head_concat_reshape)
- Created fresh Vivado project `kv260_custom_noDPU` (copied from the DPU project, then DPU/decode/nms removed)
- Wired all 9 IPs across 3 smartconnects (hp0/hp1/hp2) + 1 control smartconnect
- Hit and fixed: Zynq PS `post_propagate` TCL crash (root cause never fully isolated; fixed by deleting and recreating the PS block cleanly), HPC0 port needing a dummy AXI interconnect termination, multiple clock/reset reconnection issues
- **Synthesis + implementation succeeded** (only 5 minutes — no DPU makes routing far easier)
- Bitstream `kv260_custom_noDPU_v1.bit` deployed and loaded via `fpgautil` — but this version was missing critical connections (see Day 3)

### Day 3 — Found and partially fixed the `control_r` gap; then hit the real blocker
- **Found:** all 9 IPs had TWO separate AXI-Lite interfaces — `s_axi_control` (scalars) and `s_axi_control_r` (pointer args) — because pointer args in the `.cpp` files weren't given explicit `s_axilite` pragmas, so Vitis HLS auto-split them onto a second, unconnected bundle
- **Fixed:** expanded `smartconnect_ctrl` from 9 to 18 master ports, connected all 9 `s_axi_control_r` interfaces, reassigned addresses, rebuilt (`kv260_custom_noDPU_v2.bit`)
- Deployed v2, confirmed **all 9 IPs respond correctly on `/dev/mem`** (`ap_ctrl = 0x4 IDLE` on every IP)
- **Verified real register offsets** against Vitis-HLS-generated headers for all 9 IPs (not guessed) — confirmed 100% match with what was already coded
- Wrote full board driver (`hw_ip_driver.py`, `ip_wrappers.py`) with verified addresses
- **Systematic hardware test sequence found the real blocker:**
  1. `test_head_transpose.py` — FAIL, all-zero output
  2. `test_cma_only.py` — confirmed CMA buffer allocation itself works fine (after fixing a 32GB HPC-alias addressing bug — see below)
  3. `test_control_r_readback.py` — PASS, register writes reach the IP correctly
  4. `test_ap_ctrl_trace.py` — IP genuinely executes and completes (DONE+IDLE+READY seen)
  5. `test_sentinel.py` / `test_sentinel_v2.py` (with cache sync via `SYNC_BO` ioctl) — **output buffer never changes, even with explicit cache flush/invalidate** — ruled out cache coherency
  6. `test_pool_sentinel.py` — same IP put on a **different** smartconnect (hp1 instead of hp2) — **identical failure** — ruled out a single bad interconnect block; confirmed the problem is systemic across the whole design
- **Attempted to check PS firewall (XMPU) directly** — `/dev/mem` mmap of the XMPU register region returned `Bus error` — Linux is deliberately blocked from touching it (expected, confirms it's a trusted-firmware-controlled resource)

---

## 5. ROOT CAUSE (current best understanding)

**Not a board defect. Not a design/logic bug. Not a wiring mistake in Vivado.**

The KV260 boots PMU firmware (PMUFW) before Linux, which configures ZynqMP's hardware memory firewall (XMPU) — a security feature controlling which AXI master IDs in the PL are allowed to write to DDR. This is configured as part of Xilinx's **full Vitis platform (XSA) export flow**, which also regenerates PMUFW to whitelist each accelerator's master ID.

The original DPU-based platform on this board went through that full flow at some point (that's why DPU/decode_0/nms_top_0 work correctly). Our 9 new IPs were built through **raw Vivado only** (block design → synthesis → implementation → `write_bitstream`), skipping the platform/PMUFW regeneration step entirely. The firewall's whitelist still only reflects the old master set — our new IPs' writes are being silently blocked at the hardware level (no error surfaces to Linux; `dmesg` is completely clean; the HLS-generated AXI master logic doesn't check/report BRESP error codes, so the IP still reports "done" even though the transaction was rejected).

**Evidence supporting this conclusion:**
- Every software-visible layer checks out (control regs, pointer regs, block-design wiring, cache coherency, IP execution/completion) — 6 independent diagnostic tests, systematically eliminating each hypothesis
- Failure is 100% consistent and reproducible across 2 different IPs on 2 different smartconnects
- `/dev/mem` access to the XMPU register region is hardware-blocked from Linux (`Bus error`), consistent with PMUFW/TrustZone controlling it exclusively

---

## 6. TWO PATHS FORWARD (decision needed next session)

### Option 1 — Full platform rebuild for the no-DPU design
Export XSA from `kv260_custom_noDPU` → PetaLinux/platform rebuild → new PMUFW → new `BOOT.BIN` → reflash board.
- Pro: cleanest long-term architecture (no DPU/VART dependency at all)
- Con: entirely new toolchain never used in this project (PetaLinux), long build times (1-3 hrs), real risk of an unbootable board if misconfigured

### Option 2 — Add the 9 IPs into the EXISTING working DPU platform (chosen direction, not yet started)
Take the original DPU Vivado project (whose PMUFW is already valid) and add our 9 IPs into that same block design alongside DPU/decode_0/nms_top_0, then re-export/re-package through the same `xsct`/`v++` flow that already once produced a working `kv260_lcam.xclbin`.
- Pro: reuses a known-working PMUFW baseline (lower risk), also naturally leads to the more realistic hybrid architecture (DPU for backbone convs, custom IPs for CPU-bound ops)
- Con: `v++`/xsct packaging was already fragile in earlier attempts (several unresolved errors); still needs real work to get right

**Decision at end of Day 3: proceed with Option 2 next session.**

---

## 7. KEY FILES AND WHERE THEY LIVE

### On Windows (`C:\Xilinx\`)
```
projects\custom_ip_repo\                     — all 9 exported HLS IPs (Vivado IP catalog source)
projects\<ip_name>\                          — individual HLS project folders (9 of them)
projects\DPUCZDX8G\prj\Vivado\                — ORIGINAL DPU project (has working PMUFW/platform)
  srcs\top\top.bd                             — current block design (used for BOTH day-1 DPU+1IP
                                                 attempt AND as copy-source for noDPU project)
  xsct\kv260_lcam_ext\...\kv260_lcam_ext.xpfm — the platform used to build original working xclbin
  connectivity.cfg                            — v++ linker config from original build
projects\kv260_custom_noDPU\                  — Day 2/3 no-DPU Vivado project (copy of DPU project,
                                                 DPU/decode/nms removed, 9 IPs added)
  KV260.xpr
  KV260.runs\impl_1_01\top_wrapper.bit        — latest successfully implemented bitstream (v2)
kv260_lcam.xclbin                             — ORIGINAL working xclbin (DPU+decode+nms only) — SAFE BACKUP
kv260_custom_noDPU_v2.bit / .bit.bin          — latest no-DPU bitstream (9 IPs, loads fine, writes blocked)
backup_working_xclbin\                        — full backup of original working state
```

### On the board (`~` = `/home/root/`)
```
lcam_v5.xmodel                    — compiled model (Vitis-AI 3.0)
weights/layer_plan.json           — 153-layer execution plan
weights/weight_map.json
weights/data/*.npy                — 211 extracted weight tensors
kv260_custom_noDPU_v2.bit.bin     — currently loaded bitstream (writes blocked by firewall)
hw_ip_driver.py                   — low-level /dev/mem + zocl CMA driver (9 IPs, verified addresses)
ip_wrappers.py                    — one Python class per IP (Conv2dEngine, PoolEngine, etc.)
test_head_transpose.py            — single-IP correctness test (currently FAILS — firewall block)
test_cma_only.py                  — CMA buffer isolation test (PASSES)
test_control_r_readback.py        — register interface isolation test (PASSES)
test_ap_ctrl_trace.py             — execution/completion trace test (confirms IP runs)
test_sentinel.py / test_sentinel_v2.py — write-side isolation tests (both FAIL — this is the blocker)
test_pool_sentinel.py             — cross-check on different IP/interconnect (also FAILS — confirms systemic)
WEB09971.jpg                      — test image for eventual end-to-end inference demo
bench_e2e.py through v6           — earlier DPU-only benchmark scripts (working, stock bitstream)
```

---

## 8. VERIFIED TECHNICAL REFERENCE DATA

### AXI-Lite base addresses (verified against real Vitis HLS headers, confirmed live on board)

| IP | control (scalars) | control_r (pointers) |
|---|---|---|
| conv2d_engine_0 | 0x8000_0000 | 0x8009_0000 |
| depthwise_engine_0 | 0x8001_0000 | 0x800A_0000 |
| eltwise_add_0 | 0x8002_0000 | 0x800B_0000 |
| head_concat_reshape_0 | 0x8003_0000 | 0x800C_0000 |
| head_sigmoid_0 | 0x8004_0000 | 0x800D_0000 |
| head_transpose_0 | 0x8005_0000 | 0x800E_0000 |
| lcam_attention_gate_0 | 0x8006_0000 | 0x800F_0000 |
| pool_engine_0 | 0x8007_0000 | 0x8010_0000 |
| upsample_engine_0 | 0x8008_0000 | 0x8011_0000 |

### Board access
```
Board IP: 192.168.137.126 (static, set via `ip addr add` each boot — not persistent)
SSH: ssh root@192.168.137.126
Board hostname: xilinx-kv260-starterkit-20222 (firmware 2022.2)
Bitstream load: fpgautil -b <file.bit.bin> -f Full
```

### Important gotchas discovered (for next time)
1. **HPC alias bug**: `zocl` CMA buffers return physical addresses with a `0x8_0000_0000` (32GB) alias bit for cache-coherent access — must subtract this bit before handing the address to any non-coherent AXI master (our IPs). Fixed in `hw_ip_driver.py`.
2. **`control_r` auto-split**: any HLS IP with `m_axi` pointer args but no explicit `s_axilite` pragma on those same args gets a second, separate AXI-Lite interface that Vivado does NOT connect automatically — must be manually wired.
3. **Vivado 2022.2 `TclStackFree` crash**: intermittent synthesis crash, not a real design error — simply retry.
4. **Windows path spaces**: `C:\Users\Punnam Rahul\...` breaks Vitis HLS `add_files` — always copy source files into `C:\Xilinx\projects\<name>\` (no spaces) first.
5. **Vivado run names**: implementation run is `impl_1_01`, not `impl_1` (has been consistent across every project copy).

---

## 8B. DAY 4 ADDENDUM — Correct understanding of the firewall fix path

**Key correction from Day 3:** DPU's presence is irrelevant to the firewall/PMUFW issue. Re-adding DPU to the design (attempted briefly at start of Day 4) is unnecessary — dropped. The current `kv260_custom_noDPU` design (9 IPs, no DPU) is the right target; it does NOT need DPU added back.

**How this board actually boots (confirmed via `cat /proc/mtd` on-target):**
```
No BOOT.BIN on the SD card at all — SD card only has: Image (kernel),
boot.scr, ramdisk, system.dtb.
The REAL first-stage boot chain (FSBL+PMU+ATF+U-Boot = BOOT.BIN) lives in
SOM QSPI flash, in a Kria-specific A/B redundant scheme:
  mtd5 = "Image A (FSBL, PMU, ATF, U-Boot)"
  mtd7 = "Image B (FSBL, PMU, ATF, U-Boot)"
Confirmed via: xmutil bootfw_status → both slots currently "Bootable"
```

**The update mechanism is officially documented and genuinely safe:**
```
xmutil bootfw_update -i <path-to-BOOT.BIN>
  → writes ONLY to the currently-INACTIVE slot (A or B)
  → does not touch the working slot at all
  → reboot to try the new image
  → xmutil bootfw_update -v   <- MUST run this after successful boot to
                                  make the change permanent
  → if you DON'T run -v, or if the new image fails to boot, the board
    automatically falls back to the old working slot on next restart
Reference: https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/1641152513/Kria+K26+SOM
```

**What's actually needed to build a valid Kria `BOOT.BIN`** (this is Xilinx's own packaged flow, NOT generic bootgen with hand-picked FSBL/u-boot elfs):

| Input | Status |
|---|---|
| `custom-hardware.bit` | ✓ have it — our 9-IP no-DPU bitstream |
| `custom-metadata.xclbin` | ✗ BLOCKED — same unresolved v++/XSA packaging issue we've hit before |
| `custom-devicetree.dtsi` | ✗ NOT STARTED — describes our 9 IPs to the Linux kernel |
| `shell.json` | ✗ NOT STARTED — copy/adapt from an existing Xilinx example |

Process: clone `https://github.com/Xilinx/kria-apps-firmware` (branch matching board's tool version), place the 4 files above into `kv260/custom/`, run `make` (Yocto-based build) → produces the real `BOOT.BIN`.

**Reference docs found (Day 4 research):**
- Boot firmware overview: https://xilinx.github.io/kria-apps-docs/bootfw/build/html/docs/bootfw_overview.html
- Generating custom firmware binaries: https://xilinx.github.io/kria-apps-docs/kv260/2022.1/build/html/docs/generating_custom_firmware.html
- Kria K26 SOM wiki (A/B mechanism, `xmutil` reference): https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/1641152513/Kria+K26+SOM
- Boot Image Recovery Tool (emergency fallback if BOTH A and B ever become unbootable — requires holding FWUEN button on power-on, connecting via fixed-IP web UI at 192.168.0.111): https://xilinx.github.io/kria-apps-docs/bootfw/build/html/docs/bootfw_image_recovery.html

**Bottom line:** the actual blocker is still the same unresolved xclbin packaging step from Day 1. Fixing PMUFW/firewall recognition requires completing that xclbin build FIRST, then adding two new artifacts (device tree overlay + Yocto firmware build) on top of it. This is real, additional, multi-step work — not a quick fix.

---

## 10. DAY 5 ADDENDUM — Device tree overlay path fully built; xclbin confirmed as the true remaining blocker

**Major progress today — the proper Kria firmware packaging flow now works end-to-end:**

```
✓ bootgen built from source on-target (board has no apt/internet; used dnf-less
  approach: cloned bootgen source via WSL, scp'd to board, compiled with
  on-board gcc/g++/make — succeeded, installed to /usr/bin/bootgen)
✓ XSA exported from kv260_custom_noDPU (write_hw_platform)
✓ device-tree-xlnx repo cloned (tag xlnx_rel_v2022.2), used via XSCT/HSI to
  auto-generate pl.dtsi describing all 9 IPs — addresses matched our
  already-verified register map exactly (0x8000_0000 etc.)
✓ Manually rewrapped HSI's raw pl.dtsi into a proper device-tree OVERLAY
  fragment (/dts-v1/; /plugin/; &fpga_full{...}; &amba{ <9 IP nodes> };) —
  HSI's default output is a standalone tree, not an overlay fragment, and
  needs this wrapping to be usable by dtc for overlay compilation
✓ kria-apps-firmware repo cloned, custom-hardware.bit + custom-devicetree.dtsi
  + shell.json placed in boards/kv260/custom/
✓ make -C boards/ succeeded — produced custom-hardware.bin (bootgen) and
  custom-devicetree.dtbo (dtc) with only cosmetic warnings (same class of
  warnings seen on Xilinx's own working example apps)
✓ make -C boards/ install → registered as "kv260-custom" in
  /lib/firmware/xilinx/, appears correctly in `xmutil listapps`
✓ xmutil loadapp kv260-custom loads without error
✓ ap_ctrl register check still reads 0x4 (IDLE) after this proper load —
  IPs remain reachable via AXI-Lite exactly as before
```

**New blocker found (replaces yesterday's untested firewall theory):**

```
zocl DRM device (/dev/dri/renderD1xx + matching /dev/dri/cardN) is NOT
created when kv260-custom is loaded via xmutil loadapp — confirmed via:
  - dmesg shows zocl only ever initializes ONCE, at cold boot, for
    whichever app is the system default (kv260-benchmark-b4096)
  - Manually unbinding/rebinding the zocl-drm platform driver did nothing
    (zero bound device instances found in sysfs)
  - Rebooting cleanly and checking: default app (benchmark-b4096) DOES get
    renderD128 + card1; switching live to kv260-custom removes them and
    they do NOT reappear for our app, even fresh after boot

ROOT CAUSE IDENTIFIED: comparing installed files side-by-side —
  kv260-benchmark-b4096/ has: .bin, .bit.bin, .dtbo, .xclbin, shell.json
  kv260-custom/            has: .bin,          .dtbo,          shell.json
                                              (NO xclbin)
shell.json is byte-identical between both apps (confirmed via diff) — NOT
the differentiator. The only structural difference is the missing xclbin.

CONCLUSION: the docs' "(optional)" note on custom-metadata.xclbin applies
only to whether the FIRMWARE PACKAGING step (make/bootgen/dtc) succeeds —
it does NOT mean zocl can create a working DRM device / CMA memory pool
without one. The xclbin's mem_topology section is very likely required by
zocl at PL-load time to know how to set up the render node and buffer
allocator. This is consistent with everything found on Day 3 (before we
even knew about the proper xmutil path) and closes the loop: xclbin
packaging (the original Day 1 blocker) was never avoidable — every path
we've explored converges back to needing a working xclbin.
```

**Corrected next-session plan:** stop pursuing the device-tree-overlay-without-xclbin path further — it's a dead end for our use case (dead end ≠ wasted effort; we now have a fully working firmware packaging pipeline that will immediately pay off once the xclbin exists). The one remaining task is:

1. Build a valid `custom-metadata.xclbin` for `kv260_custom_noDPU` using `v++` (the same packaging step that blocked us on Day 1 for the DPU+lcam_attention_gate design — needs a platform `.xpfm`, which we now also know how to derive from an XSA if needed)
2. Drop that xclbin into `~/kria-apps-firmware/boards/kv260/custom/custom-metadata.xclbin`
3. Re-run `make -C boards/ && make -C boards/ install` (everything else is already proven working — bootgen, dtc, device tree overlay content, install path)
4. `xmutil unloadapp && xmutil loadapp kv260-custom`, confirm `/dev/dri/renderD1xx` now appears
5. Re-run `test_head_transpose.py` — this should be the one that finally passes

---

## 12. DAY 5 ADDENDUM PART 2 — xclbin built with MEM_TOPOLOGY + IP_LAYOUT; zocl node added to overlay; dfx-mgrd stability issue found

**Continued today — xclbin built directly via xclbinutil (bypassing the broken v++/platform route):**

```
✓ v++ platform-create route tried first, failed with "non-accelerated platform"
  error — our XSA was never marked with PFM.* properties in Vivado, so v++
  refuses to treat it as a valid Vitis platform for kernel linking. This
  path was abandoned (would require redoing the Vivado build with platform
  properties set — not worth it since we don't need kernel linking anyway).

✓ Switched to xclbinutil --add-section (direct xclbin assembly, no platform
  needed). Dumped REAL section content from the working kv260_lcam.xclbin
  to get verified schemas (not guessed):
    - MEM_TOPOLOGY: 7 entries (HPC0,HPC1,HP0-3,LPD), each with m_used,
      m_sizeKB (hex KB, e.g. 0x200000 = 2GB), m_tag, m_base_address
    - IP_LAYOUT: m_type=IP_KERNEL, m_ip_control=AP_CTRL_HS, m_base_address,
      m_name format "kernelname:instancename"
  Built matching JSON for our 9 IPs (HP0/HP1/HP2 marked used, control-
  interface base addresses from our verified register map, m_int_enable=0
  since our IPs have no interrupts wired) → successfully packaged into
  kv260_9ip.xclbin (6.68MB) via:
    xclbinutil --add-section MEM_TOPOLOGY:JSON:... 
               --add-section IP_LAYOUT:JSON:...
               --add-section BITSTREAM:RAW:top_wrapper.bit
               --output kv260_9ip.xclbin --force

✓ Deployed as custom-metadata.xclbin, rebuilt+reinstalled via
  kria-apps-firmware — still NO /dev/dri/renderD1xx after loadapp.

ROOT CAUSE #2 FOUND: dumped the WORKING benchmark app's compiled .dtbo
back to source (dtc -I dtb -O dts ...) and found a node our device tree
never had:
    zyxclmm_drm {
        compatible = "xlnx,zocl";
        status = "okay";
        interrupt-parent = <&gic>;
        interrupts = <0 0x59 4  0 0x5a 4  0 0x5b 4  0 0x5c 4
                       0 0x5d 4  0 0x5e 4  0 0x5f 4  0 0x60 4>;
    };
This is a pure SOFTWARE node (no corresponding Vivado IP) that triggers
the "[drm] Probing for xlnx,zocl" kernel message and creates the DRM
device. HSI's auto-generation never produces it because it only describes
real hardware blocks. Added this node to custom-devicetree.dtsi by hand
(inside the existing &amba{} block) — dtc compiles clean, make/install
succeeded, xclbin now has both MEM_TOPOLOGY and IP_LAYOUT.

STILL BLOCKED (new, different problem): xmutil loadapp intermittently
fails with:
    DFX-MGRD> ERROR:initSocket():374 connect(/tmp/dfx-mgrd.socket): Connection refused
    write: Transport endpoint is not connected
The dfx-mgrd background daemon appears to crash/die across repeated
unloadapp/loadapp cycles in the same session. Manually restarting it
(after `rm -f /tmp/dfx-mgrd.socket`) sometimes works for one cycle, then
dies again. A CLEAN REBOOT reliably restores it (confirmed: after reboot,
default kv260-benchmark-b4096 gets renderD128 correctly) — but our own
kv260-custom app has not yet been tested as the immediate post-reboot
load in the SAME session where dfx-mgrd was healthy from a clean boot.
```

**Corrected next-session plan:**

1. Fresh reboot FIRST — do not touch `dfx-mgrd` manually before testing
2. Immediately after boot (while `dfx-mgrd` is in its known-good freshly-started state): `xmutil unloadapp` then `xmutil loadapp kv260-custom` in one clean sequence, `ls /dev/dri/` right after
3. If `renderD1xx` finally appears: rerun `test_head_transpose.py` — this is the real test
4. If it STILL fails even on a clean boot with the complete xclbin+devicetree: the remaining candidates are (a) an error in our `IP_LAYOUT`/`MEM_TOPOLOGY` JSON that's structurally valid but semantically wrong (e.g. `m_base_address` should reference `control_r` not `control`, since that's the interface that carries data pointers), or (b) `dfx-mgrd` genuinely has a bug/limitation with FLAT/non-DFX custom shells that the pre-built example apps don't hit — worth testing whether one of the *other* working examples (e.g. `kv260-smartcam`) survives an unload/reload cycle as cleanly as `kv260-benchmark-b4096` does, to isolate whether this is app-specific or universal instability
5. Reference materials for tomorrow's professor meeting: `EXPLAINER_for_professor.md` (definitions + narrative) and the pipeline diagram, both already produced

---

## 13. IMMEDIATE NEXT STEPS (start of next session)

**(Superseded by Day 4 findings — DPU is NOT needed. Use `kv260_custom_noDPU` as-is.)**

Corrected plan for next session:

1. Fix the `custom-metadata.xclbin` packaging for `kv260_custom_noDPU` (the same v++/XSA step that blocked us on Day 1 for the DPU+lcam_attention_gate design — needs to be solved once, properly, likely by carefully following `https://xilinx.github.io/kria-apps-docs/kv260/2022.1/build/html/docs/generating_custom_firmware.html` step by step rather than ad hoc)
2. Write `custom-devicetree.dtsi` describing the 9 IPs (new artifact — look at an existing Xilinx example `.dtsi` from `kria-apps-firmware` repo for the pattern)
3. Adapt a `shell.json` from an existing example in that repo
4. Clone `https://github.com/Xilinx/kria-apps-firmware`, assemble `kv260/custom/` with all 4 files, run `make` to produce `BOOT.BIN`
5. **Backup current QSPI state first**: `dd if=/dev/mtd5 of=~/boot_image-a_backup.bin bs=1024` (belt-and-braces on top of the A/B mechanism itself)
6. `xmutil bootfw_update -i BOOT.BIN` → reboot → test → only run `xmutil bootfw_update -v` to commit once `test_head_transpose.py`/`test_pool_sentinel.py` actually PASS
7. If anything goes wrong before committing: board automatically falls back to the old working slot on next reboot — no manual recovery needed
8. Once one IP is proven correct on real hardware, move to full `layer_plan.json`-driven orchestration and the real end-to-end inference demo with `WEB09971.jpg`, comparing against the known-good DPU output (`[0.53125, 0.71875, 0.1875, 0.625, 0.0, 0.28125, 0.0625]`)

---

## 14. LATE DAY 5 FINAL UPDATE — memory bank ordering ruled out; 8 hypotheses eliminated total

```
Reordered MEM_TOPOLOGY so a used bank (HP0) sits at index 0 instead of the
unused HPC0/HPC1 — theory being that zocl's CMA allocator defaults to bank
index 0 and was hitting an unused bank, explaining the recurring dmesg
line "[drm] Allocating BO from CMA for invalid or unused memory index[0]".

Result: buffer allocation still succeeds with a valid low address
(phys_addr=0x437c000), the dmesg warning STILL appears regardless of
reordering -- meaning this message is very likely harmless/cosmetic on
this kernel version, not the actual blocker. Re-ran test_head_transpose.py
on a clean boot with the corrected xclbin: still FAIL, same all-zero
pattern as every previous attempt.

Full list of hypotheses tested and ELIMINATED today (Day 5), in order:
  1. Cache coherency (explicit SYNC_BO flush/invalidate) -- ruled out
  2. control_r register reachability -- confirmed working, ruled out
  3. IP execution/completion (ap_ctrl state trace) -- confirmed executing, ruled out
  4. CMA buffer/address validity -- confirmed valid (after HPC alias fix), ruled out
  5. AXI master block-design wiring -- confirmed connected in Vivado, ruled out
  6. Missing zocl device tree node -- found genuinely missing, added, ruled out as sole cause
  7. Missing xclbin (MEM_TOPOLOGY + IP_LAYOUT) -- built correctly, ruled out as sole cause
  8. Memory bank index ordering -- reordered, no change, ruled out

REMAINING PATH: the write transaction itself needs to be observed directly
on the AXI bus. This requires a Vivado ILA (Integrated Logic Analyzer) --
inserting debug cores into the design, resynthesizing, and capturing a
live transaction with its actual BRESP/RRESP response code. This is the
only way left to distinguish "transaction never leaves the IP" from
"transaction leaves but gets rejected downstream" -- everything visible
from software has now been checked.

NEXT SESSION: set up a Vivado ILA on one AXI master's write channel
(AWVALID/AWREADY/WVALID/WREADY/BVALID/BRESP) for head_transpose_0's
m_axi_gmem1 port, trigger on AWVALID, capture, and read the actual BRESP
value. A BRESP of 0 (OKAY) with data still not landing points to a deeper
DDR/cache issue; a BRESP of 2/3 (SLVERR/DECERR) confirms a genuine
addressing/permission rejection at the interconnect level.

---

## 15. LATE NIGHT ADDENDUM — JTAG connected and verified; ILA debug core added; synthesis running

**JTAG hardware connection established:**
```
Physical: USB-JTAG cable connected from PC to KV260 carrier board
           (board also has power supply + Ethernet connected simultaneously)
Driver:   Digilent Adept Runtime installed (was missing initially -- USB
          Serial Converters A-D showed "Unknown" status until driver
          install + reboot). Path used:
          C:\Xilinx\Vivado\2022.2\data\xicom\cable_drivers\nt64\digilent\install_digilent.exe
Verified: FTDI FT4232H chip confirmed (VID_0403/PID_6011), all 4 serial
          converter channels showing OK status post-reboot.

Vivado Hardware Manager connection CONFIRMED LIVE:
    open_hw_manager
    connect_hw_server        -> localhost:3121
    open_hw_target           -> localhost:3121/xilinx_tcf/Xilinx/XFL1IUVT4EWFA
    get_hw_devices           -> xck26_0 arm_dap_1
This confirms real, working JTAG connectivity to the KV260's FPGA chip --
a capability this project did not have until tonight.
```

**ILA (Integrated Logic Analyzer) debug core added to the design:**

```
Target signal: the AXI interconnect wire between smartconnect_hp2/M00_AXI
  and zynq_ultra_ps_e/S_AXI_HP2_FPD -- this single point sees every write
  from all 4 IPs on that interconnect (lcam_attention_gate_0,
  head_sigmoid_0, head_transpose_0, head_concat_reshape_0).

IMPORTANT GOTCHA discovered: this net's Vivado-internal NAME is
  "/smartconnect_hp0_M00_AXI" even though it is physically wired to
  smartconnect_hp2 -- a leftover label from earlier rewiring this week
  that was never renamed. Confirmed correct by querying the ACTUAL PIN
  connection rather than trusting the name:
    get_bd_intf_nets -of_objects [get_bd_intf_pins smartconnect_hp2/M00_AXI]
    -> /smartconnect_hp0_M00_AXI   (confusing name, but verified correct wire)
  IMPORTANT FOR NEXT SESSION: always resolve nets by pin, not by name
  pattern, on this design -- names are not trustworthy anymore after this
  week's repeated rewiring.

Debug marking applied via two Tcl mechanisms (both set, redundant but
harmless):
    mark_debug [get_bd_intf_nets -of_objects [get_bd_intf_pins smartconnect_hp2/M00_AXI]]
    set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets -of_objects [get_bd_intf_pins smartconnect_hp2/M00_AXI]]
Confirmed visually in Diagram view: two small green "bug" icons appeared
on the marked wire.

Tools -> Set Up Debug... wizard was NOT found in this Vivado 2022.2 GUI
layout (menu item may be named differently or located elsewhere) --
proceeded via direct Tcl marking instead, which Vivado picks up
automatically during synthesis (standard behavior: MARK_DEBUG-tagged
nets get an ILA + Debug Hub auto-inserted at synthesis time without
needing the GUI wizard).

validate_bd_design still fails with the SAME pre-existing harmless error
seen all week (.hpfm platform file / Windows path-with-spaces issue,
plus "no default platform clock selected" -- both cosmetic, never once
blocked real synthesis this week). Proceeded past it as usual.

New (expected, harmless) finding: a new smartconnect_ctrl_2 instance
appeared automatically, likely auto-created for the Debug Hub's own
AXI-Lite control connection. Shows a Low-Area Mode / WRAP-burst DECERR
warning -- worth keeping an eye on but understood as debug-hub-specific
plumbing, not related to our 9 IPs' own AXI masters.

STATUS AT SESSION END: synth_1 launched via the same proven batch script
(rebuild_full2.tcl) used successfully all week. Was still running when
the session ended for the night.
```

**Next session — exact resume point:**

1. Check `C:\Xilinx\synth_debug.log` for `SYNTH STATUS: synth_design Complete!`
2. If complete, run implementation:
   ```powershell
   & "C:\Xilinx\Vivado\2022.2\bin\vivado.bat" -mode batch -source C:\Xilinx\impl_debug.tcl -log C:\Xilinx\impl_debug.log
   ```
3. Once `IMPL STATUS: write_bitstream Complete!`, program the board via JTAG (NOT fpgautil/xmutil this time -- direct Vivado programming, since we need the live ILA connection):
   ```tcl
   open_hw_manager
   connect_hw_server
   open_hw_target
   set_property PROGRAM.FILE {C:/Xilinx/projects/kv260_custom_noDPU/KV260.runs/impl_1_01/top_wrapper.bit} [current_hw_device]
   program_hw_devices [current_hw_device]
   ```
4. Open the Hardware Manager dashboard (Window -> Hardware Manager), arm the ILA trigger:
   ```tcl
   run_hw_ila [get_hw_ilas]
   ```
5. On the board (separate terminal, same session): `python3 test_head_transpose.py`
6. Read the captured waveform -- specifically the BRESP value when BVALID
   asserts. BRESP=00 (OKAY) means the bug is deeper than the AXI bus
   (e.g. DDR controller/cache); BRESP=10/11 (SLVERR/DECERR) confirms a
   genuine rejection and tells us exactly where in the chain to look next.

Note: programming via JTAG bypasses the whole xmutil/device-tree-overlay/
zocl path we built on Day 5 -- this is intentional and fine for the ILA
capture itself (we just need the bitstream running on the chip to watch
the AXI bus), but once we have the ILA's answer, we'll need to go BACK
to the proper xmutil loadapp path (with the same debug-core-enabled
bitstream repackaged) to actually test with our Python driver, since
that's the path our zocl-based buffer allocation depends on.

---

## 16. NEXT-DAY CONTINUATION — synth+impl with debug marking completed, but ILA was never actually inserted

```
Synthesis and implementation both completed successfully with the
mark_debug-tagged net (SYNTH STATUS / IMPL STATUS both "Complete!").
JTAG programming via Vivado Hardware Manager succeeded:
    program_hw_devices [current_hw_device] -> "End of startup status: HIGH"

BUT: refresh_hw_device reported
    "Device xck26 (JTAG device index = 0) is programmed with a design
     that has no supported debug core(s) in it."
    get_hw_ilas -> "No matching hw_ilas were found."

ROOT CAUSE: mark_debug / HDL_ATTRIBUTE.DEBUG applied directly to a BLOCK
DESIGN interface net (via mark_debug on a bd_intf_net) records intent
but does NOT by itself cause Vivado to instantiate the actual ILA +
Debug Hub IP into the netlist. That requires ONE of:
  (a) The block-design "Debug" right-click flow to have actually added
      a new cell (e.g. an axi_apb_bridge/debug_bridge/ila_0 block)
      visible in `get_bd_cells` -- we never confirmed this cell existed
      after marking; likely it did NOT get added because we used the
      Tcl `mark_debug` command on the net directly rather than the
      proper GUI "right-click net -> Debug" flow (which appeared to
      select things but may not have completed the actual insertion
      dialog/confirmation).
  (b) OR: open the SYNTHESIZED design specifically (open_run synth_1,
      not the block design), apply MARK_DEBUG constraints there via
      set_property on the actual synthesized netlist nets, then run
      Vivado's proper debug-insertion flow (Tools > Set Up Debug...)
      against that OPEN SYNTHESIZED RUN -- this is the standard RTL-level
      debug flow and is likely what was actually needed; we attempted
      Set Up Debug against the block design/project level instead,
      where the menu item may not have been available for this reason.

CORRECT NEXT-SESSION PROCEDURE:
  1. open_project ... ; open_run synth_1  (NOT open_bd_design)
  2. In the resulting netlist schematic/Tcl, locate the actual signal
     name for the smartconnect_hp2<->PS AXI connection post-synthesis
     (will differ from the BD-level name, likely something under
     top_wrapper_i/smartconnect_hp0/... given the stale naming we
     already know about)
  3. set_property MARK_DEBUG true [get_nets <that path>]
  4. Tools -> Set Up Debug...  (should now be available/relevant since
     a synthesized run is open, not just a block design)
  5. Complete the wizard (accept default ILA depth/clock), this inserts
     the actual ILA + Debug Hub into the netlist
  6. write_bitstream fresh from this point, THEN JTAG program + capture

Alternative if the above is still troublesome: use the BLOCK DESIGN
"Debug" flow correctly by right-clicking the (correctly identified, by
pin not name) net in the Diagram and selecting Debug..., then WATCH for
a new cell appearing in get_bd_cells (something like debug_bridge_0 or
ila_0) before saving/regenerating -- if no new cell appears, the
insertion did not happen and needs to be retried via the GUI dialog
that pops up after right-click Debug (there should be a confirmation
dialog we may have missed/dismissed too quickly).

Current bitstream on the chip right now (via JTAG, not flash/QSPI/SD --
this program is VOLATILE and will be gone on next power cycle) is a
valid, working, no-DPU 9-IP bitstream WITHOUT a working debug core.
No harm done -- next session starts fresh on the debug-insertion step
specifically, with a clear, correct procedure now identified above.

---

## 17. MAJOR BREAKTHROUGH — Live ILA capture shows the AXI write transaction ACTUALLY SUCCEEDS at the bus level

**Full JTAG + ILA debug flow completed successfully tonight** (this itself is a real
achievement -- JTAG driver installed, connection verified, ILA core built via
Tcl (create_debug_core/create_debug_port/connect_debug_port/implement_debug_core --
the GUI "Set Up Debug" wizard was never found/used, everything done via Tcl
against an `open_run synth_1` session), debug-enabled bitstream synthesized+
implemented+programmed via JTAG, trigger armed on AWVALID, captured 1024
samples with STATUS.CORE_STATUS=FULL.

**Probes captured:** top_i/smartconnect_hp2/M00_AXI_{awvalid,awready,wvalid,
wready,bvalid,bready} -- the full write-channel handshake for the interconnect
carrying all 4 CPU-subgraph-replacement IPs (lcam_attention_gate, head_sigmoid,
head_transpose, head_concat_reshape).

**CSV data readout (samples around trigger point 512, and later window 570-924):**

```
Sample 512: awvalid=1, awready=1   <- ADDRESS HANDSHAKE SUCCEEDS
Sample 514: wvalid=1,  wready=1    <- DATA HANDSHAKE SUCCEEDS
Samples 570,575,576,577,580,613,618,623,624,625,656,661,666,671,672,
699,704,709,714,719,742,747,752,757,762,785,790,795,800,805,828,833,
838,843,848,871,876,881,886,891,914,919,924:
            bvalid=1, bready=1    <- WRITE RESPONSE ARRIVES AND IS ACCEPTED,
                                      repeatedly, roughly every 40-50 cycles
                                      across the whole 1024-sample window
```

**CONCLUSION: the AXI write transaction genuinely completes successfully at
the bus/protocol level.** Address accepted, data accepted, response returned
and acknowledged -- no DECERR, no stuck handshake, no dropped transaction
anywhere visible in the captured window. This DEFINITIVELY RULES OUT the bus/
interconnect/PS-port layer as the cause of the all-zeros failure.

**This redirects the investigation to the CPU-read side, specifically:
cache coherency was "ruled out" on Day 5 using a SYNC_BO ioctl call that was
EXPLICITLY FLAGGED AT THE TIME as an unverified, best-effort guess at the
real `drm_zocl_sync_bo` struct layout and ioctl number (fields: handle, dir,
offset, size -- guessed, never checked against actual kernel header source).
If that guess was wrong, `fcntl.ioctl()` could silently no-op (no Python
exception raised) while appearing to "work" -- meaning the earlier
cache-coherency elimination may itself be invalid and needs to be redone
with a VERIFIED ioctl definition.**

**NEXT SESSION -- concrete, specific plan:**

1. Get the REAL `drm_zocl_sync_bo` struct and `DRM_ZOCL_SYNC_BO` ioctl number
   from actual kernel source, not memory/guessing. Options:
     a. On the board: `find / -iname "*zocl*.h" 2>/dev/null` -- PetaLinux
        images sometimes ship kernel headers
     b. Check if the zocl kernel module source is available anywhere on the
        board's filesystem (/usr/src, /lib/modules/.../build)
     c. Fetch from the authoritative GitHub source used for the original
        working driver: https://github.com/Xilinx/XRT/blob/master/src/runtime_src/core/edge/include/zynq_ioctl.h
        (same repo already cited correctly for CREATE_BO/MAP_BO/INFO_BO in
        the ORIGINAL working kv260_lcam_driver_v3.py -- SYNC_BO should be
        right there in the same enum/struct block, just wasn't looked up
        carefully when we added it on Day 5)
2. Rebuild `sync_to_device`/`sync_from_device` in hw_ip_driver.py with the
   VERIFIED struct layout
3. Re-run test_sentinel_v2.py (the sentinel-value test) with the corrected
   sync calls -- if cache coherency really is the answer, the sentinel
   should finally change from 0x55 to real transposed data
4. If that STILL doesn't work, the ILA has already proven the bus/PL side
   is completely clean -- next suspects would be: physical address
   mismatch between what the PL master actually wrote to vs. what address
   the CPU is reading back from (re-verify HPC_ALIAS_BIT math is exactly
   right, or check for a second/different alias offset specific to THIS
   xmutil-loaded design vs. the raw-fpgautil-loaded design the alias fix
   was originally tested against), or a DDR controller-level QoS/ordering
   issue that requires a full memory barrier the ioctl alone doesn't cover
5. This entire JTAG+ILA setup is REUSABLE going forward -- programming via
   JTAG (open_hw_manager/connect_hw_server/open_hw_target/program_hw_devices)
   and re-arming (run_hw_ila) takes under a minute once the debug-enabled
   bitstream exists, so further capture/inspect cycles going forward will
   be fast, not another multi-hour setup

---

## 18. BOTH ADDRESSES PROVEN CORRECT -- narrows to RDATA as the last untested link

**Verified `SYNC_BO` ioctl against real Xilinx source** (https://github.com/Xilinx/XRT/blob/master/src/runtime_src/core/edge/include/zynq_ioctl.h):
our struct layout (handle/dir/offset/size) and ioctl number (DRM_COMMAND_BASE+4)
were ALREADY CORRECT, matching the real kernel header exactly. The Day 5
"unverified guess" caveat is resolved -- it was right all along. Applied
sync_to_device() after CPU writes input AND sync_from_device() before CPU
reads output (both directions, in the correct order) -- still all zeros.
**Cache coherency is now genuinely, thoroughly ruled out.**

**Extended the ILA to capture full AWADDR (write address) -- 48 bits + AWVALID/AWREADY:**
```
Captured AWADDR (sample 512, trigger) = 0xBC2000
Python out_buf.phys_addr (same run)   = 0xbc2000
RESULT: EXACT MATCH
```
The hardware writes to precisely the correct physical address.

**Extended the ILA further to capture full ARADDR (read address) -- 48 more bits + ARVALID/ARREADY:**
```
Captured ARADDR (sample 512, trigger) = 0x2750000
Python in_buf.phys_addr (same run)    = 0x2750000
RESULT: EXACT MATCH
```
The hardware reads from precisely the correct physical address too.

**Methodology note for reproducing this measurement correctly:** the ONLY
valid comparison is between an address captured by the ILA and the
Python-printed phys_addr from the SAME SCRIPT EXECUTION -- comparing
against a value printed by a separate, later Python invocation is
comparing two different buffer allocations and will always "mismatch"
meaninglessly. Created test_head_transpose_debug.py specifically to print
in_buf.phys_addr / out_buf.phys_addr inline, in the same run being captured.

**CURRENT STATE: both read and write addresses are provably, bit-for-bit
correct. AXI handshakes (AWVALID/AWREADY, WVALID/WREADY, BVALID/BREADY,
ARVALID/ARREADY all confirmed completing). Cache coherency handled
correctly on both sides. Output is STILL all zeros.**

**ONE UNTESTED LINK REMAINS: RDATA.** We have never captured the actual
DATA VALUE returned on the read channel -- only that the read address
handshake completes. The IP could be requesting the exactly correct
address and receiving a completed transaction, but if RDATA itself is
already zero (e.g. genuine race between the CPU's cache-flush completing
and the IP's read arriving, or a DRAM-level issue unrelated to addressing)
that would explain everything observed so far.

**NEXT SESSION -- exact continuation point:**
1. Mark RDATA (32 or 128 bits depending on the bus width used --
   check `get_pins "top_i/smartconnect_hp2/M00_AXI_rdata[*]"` count first)
   the SAME way ARADDR was marked (per-bit loop with get_nets -of_objects
   [get_pins ...])
2. Also mark RVALID/RREADY (same pattern as AR/AW pairs)
3. implement_debug_core -> write_checkpoint -> opt/place/route -> bitstream
   (reuse top_wrapper_debug3.dcp lineage, same ~15-20 min build)
4. Program via JTAG, write_debug_probes fresh, refresh_hw_device
5. Set trigger on RVALID this time (or keep ARVALID -- RDATA should appear
   a few cycles after AR completes, visible in the same capture window
   around sample 512-520 based on prior timing)
6. Run test_head_transpose_debug.py, capture, export CSV
7. Assemble RDATA bits the same verified way (per-bit loop, NOT manual
   eyeballing -- this session proved manual counting is error-prone;
   always use the PowerShell bit-assembly loop with correct 1-based line
   indexing, remembering Get-Content line 0 = CSV header row)
8. Compare assembled RDATA against the known INPUT data
   (src = np.arange(48) = [0,1,2,...,47] in test_head_transpose_debug.py)
   at the specific beat/sample where RVALID+RREADY are both 1
9. If RDATA shows real values (0,1,2...) -- the bug is INSIDE the IP's own
   HLS-generated compute logic (unlikely given C-sim passed, but would be
   the next place to look -- possibly an HLS pragma/interface issue
   specific to hardware vs simulation)
   If RDATA shows zeros/garbage -- the bug is a genuine DRAM/timing issue
   between the CPU's cache flush and the PL's read, suggesting a real
   memory barrier or delay is needed before triggering ap_start (e.g. add
   a small sleep after sync_to_device() before calling ip.run(), as a
   diagnostic first, before looking for a proper barrier mechanism)

**This entire JTAG+ILA infrastructure is fully proven and fast to reuse --
each additional probe-marking + rebuild + capture cycle tonight took
roughly 20-25 minutes end to end once the pattern was established.**

---

## 19. RDATA CAPTURED -- neither correct data nor zero; points at write-never-landed

**Extended ILA to capture full RDATA (128 bits, matches HP port width) + RVALID/RREADY.**
Same rebuild process (mark all 128 bits with per-bit loop -- IMPORTANT: multi-line
Tcl for-loops pasted into the Vivado Tcl console can get corrupted/reordered by
copy-paste; always use SINGLE-LINE for-loops with semicolons between statements
to avoid this, e.g.:
  `for {set i 0} {$i < 128} {incr i} { set probe_num [expr {$i + 104}]; connect_debug_port u_ila_0/probe$probe_num [get_nets -of_objects [get_pins "...[$i]"]] }`
This bit us once this session -- multi-line paste caused `$i` to run to 128
(out of range) before the loop broke; recovered by checking
`get_debug_ports -of_objects [get_debug_cores u_ila_0]` to see exactly what
succeeded before the corruption, then re-running the corrected single-line
version for just the missing probes.)

**Trigger setup gotcha:** `get_hw_probes` requires the EXACT post-implementation
probe name, which again uses the legacy `smartconnect_hp0` label despite being
wired to hp2 (e.g. `top_i/smartconnect_hp0_M00_AXI_RVALID`, all-caps for the
wider signals). Searching by substring (e.g. "RVALID") will ALSO match
"ARVALID" -- use anchored regex (`_RVALID$`) or exact full paths, not loose
substring matches, when locating probes/columns in both Tcl and the
PowerShell CSV-parsing scripts.

**Capture result:**
```
Found first sample where RVALID=1 AND RREADY=1 (the actual data-transfer beat)
Assembled RDATA (128-bit) = 0x9B347C34AA0003F5A9025BF552806014

Decoded as 16 individual bytes (little-endian, byte0 = lowest address = first
input element position):
  20, 96, -128, 82, -11, 91, 2, -87, -11, 3, 0, -86, 52, 124, 52, -101

Expected (src = np.arange(48), first 16 bytes should be):
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
```

**NEITHER match nor all-zero.** This is a NEW, DIFFERENT finding from every
previous all-zeros result -- the data on the READ channel itself looks like
essentially random/uninitialized DRAM content, not the deliberately-written
input data and not zero.

**INTERPRETATION:** This suggests the CPU's write of real input data into
`in_buf` (via `in_buf.write_array(src)` + `in_buf.sync_to_device()`) may
genuinely not have physically landed in DRAM by the time the IP's read
transaction executed -- despite the sync_to_device() call being verified
correct against real kernel source. This is DIFFERENT from a stale-cache
explanation (which would show OLD but real data, or zeros if never written
at all in a previous run) -- genuinely-random-looking bytes suggests either:
  (a) truly uninitialized/never-written memory (the CMA allocator gave us a
      "fresh" region that inherently starts as garbage, and our write
      genuinely never made it there before the read), or
  (b) we captured the WRONG beat/sample (there may be multiple RVALID+RREADY
      beats in this burst -- we only checked the FIRST one found; the script
      should be modified to print ALL matching beats, not just the first,
      to rule out beat-ordering confusion)

**NEXT SESSION -- immediate next steps:**
1. Modify the PowerShell extraction script to loop through the ENTIRE
   capture and print EVERY sample where RVALID+RREADY=1 (not just the
   first), decode RDATA at each, and check if ANY of them contain
   recognizable values from the real input (0-47) -- this rules out (b)
2. If truly no beat contains real data: add a diagnostic delay (e.g.
   `time.sleep(0.01)` in test_head_transpose_debug.py between
   `in_buf.sync_to_device()` and `ip.run(...)`) to test whether this is a
   genuine timing race between the CPU's cache-flush ioctl RETURNING and
   the underlying physical DRAM write actually COMPLETING -- if adding a
   delay fixes it, this points to needing a genuine memory barrier /
   different sync mechanism, not just the ioctl call existing
3. Consider testing with a plain, non-CMA reserved-memory region accessed
   directly via /dev/mem (bypassing zocl's BO/CMA abstraction entirely) as
   an isolation test -- if that has the same problem, the issue is at the
   ARM cache/memory-system level generally, not specific to zocl's CMA
   implementation
4. This entire JTAG+ILA capture infrastructure remains fully working and
   fast (~20-25 min per probe-set change, program+capture cycle itself
   under 2 minutes) -- reuse directly, no rebuilding from scratch needed

---

## 20. LIKELY ROOT CAUSE FOUND -- CPU/PL cache coherency requires the HPC port, not HP0/1/2

**Isolation test: does the CPU see its own write correctly (zero FPGA involvement)?**
```python
buf.write_array(src); buf.sync_to_device()
readback_no_sync = buf.read_array(...)   # WITHOUT sync_from_device
readback_synced  = buf.read_array(...)   # WITH sync_from_device
```
BOTH readbacks matched the real data (0..47) PERFECTLY. The CPU is 100%
self-consistent -- it can always correctly read back what it itself wrote,
with or without the sync call. This rules out any bug in the CPU-side
mmap/buffer mechanism itself.

**Combined with the RDATA finding (section 19): the FPGA's AXI read of the
EXACT SAME physical address, at the exact same point in time, returns
garbage -- not zero, not the real data, genuinely uninitialized-looking
content.**

**This is the classic signature of an ARM cache-coherency DOMAIN mismatch:**
a cache-flush/clean operation can be visible to "Point of Unification" (other
CPU cores) without being visible to "Point of Coherency" (external,
non-coherent DMA masters like a PL AXI port). CPU self-reads only ever test
the former; FPGA reads test the latter.

**Direct evidence supporting this:** tested the previously-unused
`DRM_ZOCL_BO_FLAGS_COHERENT` flag (0x1 << 27, from the real zynq_ioctl.h
header fetched this session) combined with CMA:
```
flags = DRM_ZOCL_BO_FLAGS_CMA | DRM_ZOCL_BO_FLAGS_COHERENT
CREATE_BO succeeded, phys_addr = 0x806485000
```
**This physical address is in the SAME +32GB alias range** that was
identified and stripped off back on Day 5 (the "HPC_ALIAS_BIT" fix,
0x8_0000_0000). This is not a coincidence -- that high-address alias is
specifically how ZynqMP exposes DRAM through the cache-COHERENT HPC ports
(S_AXI_HPC0_FPD / S_AXI_HPC1_FPD), architecturally separate from the
non-coherent HP ports (S_AXI_HP0/1/2_FPD) that ALL 9 of our custom IPs are
currently wired to in the block design.

**LIKELY CONCLUSION: true CPU<->PL cache coherency on this platform requires
routing through S_AXI_HPC0/1_FPD, not S_AXI_HP0/1/2_FPD.** Our IPs sit on
the wrong port class for automatic coherency; the manual cache-flush ioctls
we've been using (correctly implemented per the real kernel header) are
evidently insufficient to bridge that specific gap for this SoC/kernel
combination when the far end is a non-coherent port.

**IMPORTANT HEAD START ALREADY IN THE DESIGN:** `kv260_custom_noDPU`
already contains an `axi_interconnect_hpc0` block, added earlier purely as
a dummy termination to satisfy Vivado's requirement that the HPC0 port not
be left dangling (see section 8, PS reconfiguration notes). This is a real,
existing entry point to test the coherent-port theory directly, without
needing to invent new wiring from scratch.

**NEXT SESSION -- concrete test plan:**
1. Pick ONE small IP (e.g. head_transpose, our existing test subject) and
   re-wire ONLY its m_axi data ports to route through the existing
   axi_interconnect_hpc0 -> S_AXI_HPC0_FPD path instead of smartconnect_hp2
   -> S_AXI_HP2_FPD
2. Update its control_r/control addresses if the address map shifts as a
   result of the new interconnect assignment (re-check via Address Editor)
3. Rebuild (synth+impl+bitstream -- same proven fast no-DPU flow)
4. Update hw_ip_driver.py's ZoclBuffer to use
   DRM_ZOCL_BO_FLAGS_CMA | DRM_ZOCL_BO_FLAGS_COHERENT for buffers used with
   this specific IP, and DO NOT strip the +32GB alias bit for these
   buffers (only strip it for buffers destined for the non-coherent
   HP0/1/2 IPs, if any remain)
5. Re-run test_head_transpose.py against this one IP -- if the output
   finally shows real transposed values, the coherent-port theory is
   CONFIRMED, and the remaining 8 IPs need the same re-wiring treatment
   (may require consolidating fewer, larger AXI masters onto the limited
   number of available HPC ports -- ZynqMP typically exposes only 1-2 HPC
   ports total, far fewer than the 3 HP ports currently spread across 9
   IPs, so some AXI multiplexing/arbitration redesign will likely be
   needed for the full 9-IP system)
6. If this does NOT fix it either: the JTAG+ILA infrastructure remains
   available to directly probe the HPC0 port's AXI signals the same way
   we did for HP2 tonight, to keep narrowing further

---

## 21. COHERENT-PORT HYPOTHESIS TESTED AND RULED OUT

**Rewired head_transpose_0's data ports (m_axi_gmem0/1) from smartconnect_hp2
(non-coherent S_AXI_HP2_FPD) to the existing axi_interconnect_hpc0 block
(coherent S_AXI_HPC0_FPD):**
```
- Expanded axi_interconnect_hpc0 to NUM_SI=2
- delete_bd_objs on old gmem0/gmem1 nets, reconnected to S00_AXI/S01_AXI
- Connected S01_ACLK/S01_ARESETN (S00 already had clock/reset from its
  earlier dummy-termination setup)
- assign_bd_address initially failed ("no available apertures" / stale
  HP2_DDR_LOW segment still attached) -- fixed by explicitly deleting the
  old address segments first:
    delete_bd_objs [get_bd_addr_segs -of_objects [get_bd_addr_spaces head_transpose_0/Data_m_axi_gmem0]]
  then manually creating the new segment (automatic assign_bd_address
  would not offer HPC0_DDR_LOW even after the old segment was cleared --
  had to use create_bd_addr_seg explicitly):
    create_bd_addr_seg -range 0x20000000 -offset 0x00000000 [get_bd_addr_spaces head_transpose_0/Data_m_axi_gmem0] [get_bd_addr_segs zynq_ultra_ps_e/SAXIGP0/HPC0_DDR_LOW] SEG_gmem0_HPC0_DDR_LOW
- CONFIRMED: head_transpose_0's CONTROL addresses (0x80050000/0x800e0000)
  are UNCHANGED -- only the DATA path moved. Existing device tree overlay's
  control addressing remains valid; only the underlying bitstream changed.
```

**Rebuilt successfully** (synth+impl+bitstream, survived a mid-session
laptop crash cleanly -- block design changes were saved to top.bd before
the crash and survived; only the in-progress synth_1 run needed restarting).

**Deployed via the proven kria-apps-firmware/xmutil pipeline** (copied new
top_wrapper.bit as custom-hardware.bit, `make -C boards/ install`,
`xmutil unloadapp && xmutil loadapp kv260-custom` -- loaded cleanly, same
harmless warnings as always).

**Test with DRM_ZOCL_BO_FLAGS_CMA | DRM_ZOCL_BO_FLAGS_COHERENT, address
alias bit NOT stripped (self-contained test_head_transpose_coherent.py,
independent of hw_ip_driver.py to isolate the test):**
```
in_buf.phys_addr  = 0x8065b3000  (high alias range, as expected for COHERENT)
out_buf.phys_addr = 0x803bcb000
HW call completed in 0.0714 ms   <- notably slower than usual ~0.013-0.015ms,
                                     consistent with genuinely routing through
                                     a different (coherent) interconnect path
HW result: all zeros              <- SAME FAILURE PATTERN AS EVERY PREVIOUS TEST
```

**CONCLUSION: the coherent-HPC-port hypothesis is RULED OUT.** The new
routing is confirmed genuinely active (different completion timing proves
it's not just running the old bitstream), but produces the identical
all-zeros result as the non-coherent path. This was a well-reasoned,
evidence-supported hypothesis (matching alias-address behavior) that did
not pan out -- a real, valuable elimination, not a wasted effort.

**COMPLETE LIST OF ELIMINATED HYPOTHESES (as of this session):**
1. Cache coherency via SYNC_BO ioctl -- ruled out, verified against real kernel source
2. control_r register reachability -- confirmed working
3. IP execution/completion -- confirmed executing (ap_ctrl trace)
4. CMA buffer/address validity -- confirmed valid
5. AXI master block-design wiring -- confirmed connected
6. Missing zocl device tree node -- found and fixed, not sole cause
7. Missing xclbin metadata -- built correctly, not sole cause
8. Memory bank index ordering -- ruled out
9. Write address (AWADDR) correctness -- PROVEN exact match via ILA
10. Read address (ARADDR) correctness -- PROVEN exact match via ILA
11. Full AXI protocol completion (AW/W/B and AR/R channels) -- PROVEN via ILA
12. Beat-ordering confusion in RDATA capture -- ruled out (checked all 3 beats)
13. Timing race (100ms artificial delay) -- ruled out, no effect
14. CPU-side self-consistency -- CPU always correctly reads back its own writes
15. Non-coherent vs coherent AXI port routing -- TESTED, ruled out this session

**GENUINELY REMAINING CANDIDATES (none yet tested):**
- An SMMU (System MMU) or other address-translation layer that could cause
  the SAME NUMERIC physical address to resolve to DIFFERENT underlying DRAM
  cells for CPU-issued vs PL-issued transactions, despite our ILA proving
  bit-identical AWADDR/ARADDR values -- this would explain a match at the
  address level while still landing in genuinely different memory
- A kernel/zocl/PetaLinux-build-specific DMA-mapping bug particular to this
  board's exact software stack, independent of our design
- Something in the ARM cache architecture beyond simple flush/invalidate --
  e.g. write-allocate policy interactions, speculative prefetch, or a
  cache line still being "in flight" through an intermediate buffer that
  neither COHERENT flag nor manual sync fully addresses on this specific
  SoC/kernel/PetaLinux combination

**SESSION LENGTH NOTE:** this investigation session ran for an exceptionally
long, continuous duration (spanning late night into early morning across
multiple work blocks). Real, substantial, well-documented progress was made
-- a fully working JTAG+ILA debug infrastructure now exists and is reusable
in minutes; 15 distinct hypotheses were tested and eliminated with hard
evidence for each, which is genuinely strong methodology. The remaining
candidates require either external SMMU/kernel-level investigation or
fresh eyes -- a good, deliberate stopping point rather than continuing to
iterate while fatigued.

---

## 22. IMPORTANT DISCOVERY -- two SEPARATE System RAM regions, not one aliased region

**SMMU/IOMMU check (quick, cheap, safe -- done with remaining time):**
```
ls /sys/class/iommu/  -> EMPTY, no IOMMU groups registered anywhere
```
**SMMU/IOMMU address-translation hypothesis is CLEANLY RULED OUT** -- no
translation layer is active on this system at all.

**While checking this, found something much bigger in the full /proc/iomem:**
```
00000000-7fefffff   : System RAM     <- low region, ~2GB (0 to ~2GB)
...
800000000-87fffffff : System RAM     <- a SEPARATE declared region, at 32GB+
  87b000000-87f5fffff : reserved
  87f685000-87f744fff : reserved
  (several more reserved sub-blocks within the high region)
```

**These are TWO DISTINCT "System RAM" declarations in the kernel's own
memory map -- not one physical DRAM aliased at two addresses.** This
directly challenges the assumption made on Day 5 (section on "HPC alias
bug"): back then, a CMA buffer's phys_addr came back as e.g. 0x80130c000,
and subtracting 0x800000000 gave a low, "sane-looking" address
(0x130c000) that the CPU could successfully read/write. This was
interpreted as "stripping a coherent-access alias to get the true
address." But given /proc/iomem shows BOTH ranges as independently
declared System RAM, it's equally possible that:
  0x80130c000 and 0x130c000 are two DIFFERENT, UNRELATED physical DRAM
  locations (both genuinely valid, backed by real RAM) -- meaning every
  time we "fixed" a non-coherent-port buffer's address by subtracting
  0x800000000, we may have redirected the hardware's read/write target
  to a real-but-WRONG piece of memory, completely unrelated to the actual
  page backing the CPU's mmap'd buffer (which stays at the TRUE high
  address the whole time, e.g. 0x80130c000).

This would explain the entire pattern seen throughout Days 5 and today
PERFECTLY for the non-coherent-port (HP0/1/2) tests: the AXI transaction
completes cleanly (writing to SOME valid low-address RAM location, hence
no bus errors, hence ILA shows perfect handshakes), the captured
AWADDR/ARADDR match Python's phys_addr EXACTLY (because Python computed
that same wrong low address and told both the hardware AND itself to use
it -- self-consistent but potentially pointing at the wrong physical
page entirely, disconnected from where the CPU's OWN buffer actually
lives).

**IMPORTANT CAVEAT: today's coherent-port test (section 21) used the
FULL, UN-STRIPPED high address throughout (0x8065b3000, matching the
`800000000-87fffffff System RAM` region exactly) for BOTH the CPU buffer
AND the value written into the IP's control_r registers -- and it STILL
produced all-zeros. This means the two-separate-RAM-regions theory does
NOT by itself explain TODAY's coherent-port failure. It may still fully
explain the EARLIER non-coherent-port failures (Days 5 through most of
today), which is itself a significant, previously-unknown compounding
bug layered on top of whatever is still wrong with the coherent path.**

**NEXT SESSION -- HIGHEST PRIORITY, test this first:**
1. Re-test a NON-COHERENT-port IP (e.g. revert head_transpose to
   smartconnect_hp2, or pick a different IP still on HP0/1/2) using the
   FULL, un-stripped high address (0x800000000+) instead of the
   alias-stripped low address -- i.e. temporarily comment out or bypass
   the `HPC_ALIAS_BIT` subtraction in hw_ip_driver.py's ZoclBuffer class
   and see if a non-coherent-port IP suddenly works correctly when given
   the TRUE address instead of the "corrected" one
2. If that works: the Day 5 "HPC alias fix" was actually a bug we
   introduced ourselves, not a fix -- every non-coherent IP needs the
   full un-stripped address, and this single one-line change could
   resolve the entire investigation for 8 of the 9 IPs immediately
3. Separately, continue investigating why the COHERENT-port path (which
   already correctly uses the full un-stripped address) still fails --
   this remains a genuinely open question requiring either further ILA
   capture on the HPC0 interconnect specifically, or examination of
   whether axi_interconnect_hpc0's own configuration (data width,
   protocol conversion settings) might not be fully compatible with the
   burst/protocol behavior our HLS-generated AXI masters use
4. Full physical memory map for reference (from /proc/iomem on-target):
   Low RAM: 0x00000000 - 0x7FEFFFFF (~2GB)
   High RAM: 0x800000000 - 0x87FFFFFFF (~2GB), with several reserved
   sub-blocks near the top (0x87B000000 and above) -- CMA allocations
   from zocl with the COHERENT flag land in this high region; CMA
   allocations WITHOUT the COHERENT flag apparently land in the high
   region too (same raw ioctl always returns a high address) but were
   being manually corrected down into the low region by our own code

---

## 23. PRAGMA/CONTROL_R FIX -- CLEANLY TESTED AND RULED OUT (properly controlled)

**Compared against the one proven-working IP in the project (decode.cpp, from
the original DPU build) and found a real architectural difference:**
```
Working decode.cpp:  #pragma HLS INTERFACE s_axilite port=in_data  bundle=control
                      #pragma HLS INTERFACE s_axilite port=out_data bundle=control
                      (pointer args explicitly bundled with scalars -> ONE interface)

Our IPs (all 9):      no s_axilite pragma for pointer args at all
                      -> Vitis HLS auto-split into control + control_r (TWO interfaces)
```

**Rewrote head_transpose.cpp to match the working pattern exactly.** Verified
at every level:
```
✓ C-sim passed (3/3 tests, logic unchanged)
✓ C-synthesis interface report: exactly ONE S_AXILITE interface (was two)
✓ Exported component.xml: confirmed no s_axi_control_r anywhere
✓ upgrade_ip in Vivado: cleanly removed control_r port/interface/address block
✓ report_ip_status: Up-to-date after upgrade
```

**First test attempt was CONFOUNDED:** after upgrading the IP, discovered
head_transpose_0 was STILL wired through axi_interconnect_hpc0 (yesterday's
coherent-port experiment, section 21) -- never reverted. This meant the
first "failure" tested pragma-fix+coherent-port combined, not the intended
pragma-fix+original-hp2-port. Confirmed via block design load log:
"Excluding slave segment .../SAXIGP0/HPC0_LPS_OCM from address space
/head_transpose_0/..." (SAXIGP0=HPC0, should have been SAXIGP4=HP2).
ILA also confirmed this: armed on smartconnect_hp2, waited, NEVER triggered
(STATUS.CORE_STATUS = "WAITING FOR TRIGGER") -- because traffic was
actually flowing through the untested HPC0 wire instead.

**Properly reverted head_transpose_0 back to smartconnect_hp2** (deleted
stale HPC0 connections/address segments, reconnected m_axi_gmem0/1 to
smartconnect_hp2 S05_AXI/S06_AXI, reassigned addresses -- confirmed
HP2_DDR_LOW assigned correctly this time, not HPC0). Rebuilt clean
(synth_design Complete!, write_bitstream Complete!, 0 errors).

**Also created the CORRECT test script** (test_head_transpose_v2.py) using
the REAL post-fix register map read directly from the generated
xhead_transpose_hw.h (ap_ctrl=0x00, data_in=0x10, data_out=0x1c, H=0x28,
W=0x30, C=0x38 -- all on the single control interface, verified against
actual synthesized IP, not guessed) -- the earlier test_head_transpose.py
was stale (still targeting the old two-interface layout) and its "FAIL"
result was meaningless.

**PROPERLY CONTROLLED FINAL TEST RESULT:**
```
Real pragma fix (single control interface, matches working decode.cpp) +
Real smartconnect_hp2 routing (original, non-coherent port) +
Real verified addresses (from actual generated header) +
Correct test script (matching real register map)
=
STILL ALL ZEROS. Same failure pattern as every single test since Day 5.
```

**CONCLUSION: the control_r/interface-architecture hypothesis is CLEANLY
RULED OUT.** This was the most concrete, best-evidenced lead of the entire
investigation (direct comparison against a proven-working reference), and
it does not explain the root cause. This is genuine, valuable elimination.

**COMPLETE UPDATED LIST -- 16 hypotheses now tested and eliminated:**
(all 15 from section 21, plus:)
16. HLS interface architecture (control_r split vs unified control bundle) -- ruled out

**GENUINELY REMAINING AVENUES for a fresh session:**
- Compare the actual GENERATED RTL/loop structure between decode.cpp and
  our IPs beyond just the pragmas -- e.g. burst behavior, pipeline
  initiation interval, how each handles the AXI master's internal FSM.
  decode.cpp may have structural RTL differences beyond the interface
  pragmas that aren't visible from the .cpp source alone -- would need to
  diff the actual generated Verilog/VHDL between a working decode_0
  instance and our IPs.
- Consider building ONE new, minimal, from-scratch test IP that is as
  close to byte-for-byte identical to decode.cpp as possible (not just
  matching pragmas, but matching overall coding style/structure too) to
  see if THAT works, which would isolate whether something about HOW our
  IPs' C++ is written (independent of pragmas) produces different HLS
  scheduling/RTL behavior.
- Revisit whether decode_0/nms_top_0 (the ORIGINAL working IPs) still
  actually work TODAY, on THIS current board state, as a sanity check
  that the board/kernel/zocl stack itself hasn't developed a NEW problem
  independent of anything in our own design (would require briefly
  reloading a design that includes decode_0, e.g. from a backup xclbin,
  which is a bigger detour but would be a valuable isolation test)

---

## 24. RDATA RE-CAPTURE ATTEMPT -- NEW FINDING: the debug-instrumented bitstream itself crashes the whole board (not just returns bad data)

**Goal for this session:** re-run the §19 RDATA capture on the CURRENT
bitstream per PROJECT_CONTEXT.md's "Suggested next steps" #2 -- check every beat
across the full 1024-sample window this time, not just the first
RVALID&RREADY beat.

**Environment note (update PROJECT_CONTEXT.md):** the board's SSH root password is
now `<BOARD_PASSWORD>`, NOT `root`/`root` as PROJECT_CONTEXT.md documents -- confirmed
changed at some point since that doc was written. Everything else in
PROJECT_CONTEXT.md's environment section (board IP, the need to re-run `ip addr
add`/`ip route add` after every reboot) was reconfirmed accurate.

**Vivado/JTAG debug-core work -- successfully extended the existing
u_ila_0 core (originally just the 6 write-channel probes from §17) to
136 probes (128 RDATA bits + RVALID + RREADY), via the confirmed-working
explicit bootstrap:**
```
create_debug_core u_ila_0 ila
set_property C_DATA_DEPTH 1024 / C_TRIGIN_EN false / C_TRIGOUT_EN false /
             C_ADV_TRIGGER false / C_INPUT_PIPE_STAGES 0 /
             C_EN_STRG_QUAL false / C_CLK_INPUT_FREQ_HZ 100000000
             [get_debug_cores u_ila_0]
connect_debug_port u_ila_0/clk [get_nets top_i/zynq_ultra_ps_e/pl_clk0]
# then per-bit: create_debug_port u_ila_0 probe ; connect_debug_port
# u_ila_0/probeN [get_nets -of_objects [get_pins "...[$i]"]]
```
(This is now saved as `vivado_scripts/rdata_recapture.tcl`.)

**Two new gotchas found and added to `vivado_scripts/README.md` (#7) /
this script:**
1. **Debug-port editing (`create_debug_port`/`connect_debug_port`)
   requires `open_run synth_1`, NOT `open_run impl_1_01`.** Attempting it
   against the implemented (placed+routed) design fails with `[Vivado
   12-4097] This Vivado Debug command cannot be performed on the current
   design`. Debug-core state does NOT necessarily read the same way
   between `synth_1` and an implemented run of it -- re-check fresh on
   whichever is actually open, don't assume continuity.
2. **`get_debug_ports -of_objects [get_debug_cores u_ila_0]` includes
   `u_ila_0/clk` in its count**, alongside every `probeN` -- naively
   using `llength` on the raw list is off-by-one for computing the next
   auto-assigned probe index. Filter for `"*probe*"` first
   (`lsearch -all -inline -glob $all_ports "*probe*"`). Hit this twice in
   one session before it was caught both times by independently
   cross-checking the arithmetic before trusting it.

Also hit and resolved: `pscp` (unlike `plink`) needs its own explicit
`-batch`/`-hostkey` flags or it hangs on an unanswerable prompt; and this
board's minimal image has no SFTP server, so `pscp` needs `-scp` to force
the legacy SCP protocol or it fails with `sh: /usr/libexec/sftp-server:
No such file or directory`.

**Rebuild succeeded cleanly** (0 errors, 50 infos, 304 warnings), and the
JTAG program+arm sequence was independently verified working multiple
times across the session (136 probes confirmed live via
`get_hw_probes`, `CORE_STATUS: WAITING FOR TRIGGER` confirmed via a
live, non-cached property read each time).

**THE ACTUAL NEW FINDING:** every attempt to run `test_head_transpose_debug.py`
(the real HW call, `ip.run()` -> `AP_START` -> the actual AXI master
transaction) against this 136-probe debug-instrumented bitstream **crashed
the entire board** -- not just bad data, a full reboot. Confirmed via:
identical `plink` error signature both times ("Software caused connection
abort" / "Connection timed out"), total ping dropout, and `uptime`
showing a fresh 2-4 minute boot immediately after each recovery. This
happened **4 times** across the session (1 ambiguous, triggered
accidentally via a background-launch shell-quoting mistake; 3 clean,
deliberate, foreground runs with identical outcome). Full power-cycle +
manual `ip addr`/`ip route` re-application was required to recover the
board every time; the board itself was otherwise completely healthy
between attempts (SSH, `xmutil`, `dmesg`, file transfer, and an isolated
buffer-only precheck -- allocate/write/sync with NO IP call -- all worked
cleanly and repeatably).

**This is qualitatively different and more severe than anything else in
this entire investigation.** Every prior test (16+ hypotheses, dozens of
runs) against the ORIGINAL (non-debug) bitstream only ever produced
silent all-zeros or garbage RDATA -- never once crashed the board itself.
Only this debug-instrumented build has done that.

**Crash point was NOT consistent run-to-run** (checked via a
purpose-built `debug_scripts/test_head_transpose_pinpoint.py`, which
prints+flushes after every individual step instead of just at start/end):
- 2 of 3 clean runs: died somewhere AFTER both `phys_addr` values printed
  (i.e. after buffer allocation succeeded) -- narrows to
  `write_array`/`sync_to_device`/the register writes/`AP_START`/the
  `poll_ap_done` loop, but not more precisely than that band, since the
  pinpoint script's own first checkpoint bundles both `ZoclBuffer()`
  calls under one step (a real gap in that script, not yet fixed).
- 1 of 3 clean runs: died BEFORE printing either `phys_addr` value --
  i.e. during the very first buffer allocation itself, a step that has
  never failed anywhere else in this project's history and does not
  touch head_transpose's registers at all.

**Working hypothesis (not yet confirmed):** inconsistent, run-to-run-varying
failure points are a classic signature of a MARGINAL TIMING violation
rather than a deterministic logic bug -- i.e. inserting 136 extra debug
probes into the design may have pushed some AXI-adjacent path (on
`smartconnect_hp2` or nearby) past its timing constraint without Vivado
flagging an outright implementation error, and the resulting
metastability/glitch manifests differently depending on exact runtime
timing each time. This has NOT been verified -- see next steps below.

**TIMING HYPOTHESIS TESTED AND RULED OUT (same session):** ran
`report_timing_summary -delay_type min_max` against the live, routed
debug design (`current_design` = design_1, run status "write_bitstream
Complete!"):
```
WNS (setup) = +0.165 ns   -- MEETS timing
WHS (hold)  = +0.009 ns   -- MEETS timing
```
Both positive, so the 136 added debug probes did NOT push any path past
its constraint. **Marginal timing does not explain the crashes.**

Two caveats on that conclusion:
- WHS = +0.009 ns is a razor-thin 9 ps of hold margin. Technically
  passing and Vivado signs off, but it is not a comfortable margin and
  is worth remembering if this area comes up again.
- The follow-up `report_timing -from [get_pins -filter {NAME =~
  "*smartconnect_hp2*" || NAME =~ "*u_ila_0*"}]` returned "No paths
  found" -- that is the FILTER failing to match (probe connections
  aren't pin startpoints, and those nets carry the stale `hp0` label
  anyway), NOT evidence that no problematic paths exist. There is
  therefore still no path-level timing detail specific to the probes or
  that interconnect; only the design-wide headline numbers above.

**NOT yet tried / genuinely next** (reordered -- the original #1 was the
timing check, now done and ruled out above):
1. Rebuild with a MUCH smaller probe set (e.g. 16 RDATA bits at a time
   across several smaller builds, reconstructing the full 128-bit value
   across multiple safer capture runs) instead of all 128+6 at once, to
   see if crash frequency correlates with probe count. NOTE: the
   timing-margin rationale for this is now gone, but it remains worth
   trying on the weaker "added routing congestion / debug-hub load
   affects something not captured by static timing" theory -- treat it
   as a lower-confidence lead than it was before the timing result.
2. Try adding a short settle delay after `program_hw_devices`/
   `refresh_hw_device` before triggering any DMA/HW-call activity, and
   see if that changes crash frequency or timing (the one run that
   crashed earliest, during buffer allocation itself, was also the one
   run with the least elapsed time between JTAG programming and the test
   script executing).
3. Investigate the JTAG-programming path itself as the variable, rather
   than the probes: every crash this session followed a `program_hw_devices`
   JTAG load, whereas the entire prior history of "returns zeros but
   never crashes" was against bitstreams loaded via xmutil/fpgautil.
   Worth testing whether the ORIGINAL (non-debug) bitstream ALSO crashes
   the board when loaded via JTAG instead of xmutil -- that would shift
   the cause from "the debug probes" to "JTAG-loaded configuration vs.
   the OS-managed load path" (e.g. clocks/resets/isolation that xmutil's
   dfx-mgr sets up but raw JTAG programming does not). This is a clean,
   cheap, high-information experiment and is probably the single best
   next hardware test.
4. **Lower-risk alternative that sidesteps all of the above:** pursue
   PROJECT_CONTEXT.md's suggested-next-step #3 (sanity-check whether ANY custom
   AXI master can correctly read/write DRAM right now) using the
   ORIGINAL, stable, non-debug bitstream with a DIFFERENT one of the 9
   IPs (e.g. `pool_engine` or `eltwise_add`) -- this needs zero further
   debug-core work, carries none of today's crash risk, and would still
   produce useful information (confirming or ruling out a board-wide
   regression independent of head_transpose specifically).

---

## 25. ROOT CAUSE OF THE CRASHES FOUND -- it is the JTAG LOAD PATH, not the debug probes (next session, 2026-08-29)

Ran the two experiments proposed in §24 (#3 then #1). Result: **§24's "next
step #3" hypothesis is CONFIRMED, and the probe-count theory is dead.**

**Experiment A -- baseline, xmutil-loaded original design:**
```
xmutil unloadapp ; xmutil loadapp kv260-custom   -> "loaded to slot 0"
/dev/dri/renderD128 present
python3 test_head_transpose_debug.py
  -> in_buf.phys_addr = 0x2271000 / out_buf.phys_addr = 0x2355000
  -> HW call completed in 0.0135 ms
  -> HW result: [0, 0, 0, ... 0]   (all zeros -- the ORIGINAL bug)
  -> EXIT clean, uptime CONTINUOUS (no reboot)
```
The familiar all-zeros bug reproduces exactly as it always has, and the
board does NOT crash. Baseline healthy.

**Experiment B -- same test, but bitstream JTAG-programmed instead:**
Programmed `KV260.runs/impl_1_01/top_wrapper.bit` via `program_hw_devices`.
IMPORTANT INCIDENTAL FINDING: that file is NOT a clean non-debug bitstream
-- `get_hw_ilas` reports 1 ILA core in it, and the probes-file mismatch
warning ("port index 133 ... does not exist in the ILA core") shows it is
an OLDER, SMALLER debug build (§17-19 lineage), not yesterday's 136-probe
one. Checking this before running the test is what kept the experiment
meaningful -- ALWAYS verify with `puts "ILA CORES: [llength [get_hw_ilas
-quiet]]"` after programming, never assume a .bit is the build you think.
```
Board survived the JTAG programming itself (uptime continuous, 34 min).
python3 test_head_transpose_debug.py
  -> both phys_addr lines printed
  -> then DEAD. ping fails, SSH aborts, full reboot required.
```

**THE DISCRIMINATOR:**
| Config | Load path | Probe count | Crash? |
|---|---|---|---|
| Original packaged app | xmutil | none | **NO** |
| Older/smaller ILA build | JTAG | ~6 | **YES** |
| §24's debug build | JTAG | 136 | **YES (4/4)** |

**Probe count is definitively NOT the variable** -- a ~6-probe build
crashed exactly as reliably as the 136-probe one. 5 crashes on
JTAG-loaded designs, 0 crashes on xmutil-loaded, clean split.

(Caveat on the §24 reasoning this overturns: §24 argued the project's own
§17-19 history showed JTAG+ILA+this test working fine, which weakened the
JTAG hypothesis. That reasoning was wrong, or §17-19's setup differed in
some way not captured in the write-up -- today's direct test beats the
inference from history.)

**LIKELY MECHANISM (strong hypothesis, mechanism not yet directly proven):**
`xmutil`/dfx-mgr does far more than push a bitstream into the fabric -- it
applies the device tree overlay, configures clocks/resets, and registers
the design's IP layout with `zocl`. Raw `program_hw_devices` ONLY rewrites
the PL fabric. After a JTAG program, the OS's model of the hardware
(address map, IP layout, zocl registration, DT) can disagree with what is
physically in the PL. When the CPU then writes to `0x80050000` expecting
head_transpose and **no slave responds at that address**, the AXI
transaction never completes -- and an unanswered AXI access on ZynqMP
hangs the interconnect, tripping the watchdog into a full reboot.

This fits every observation, including the timing of death within the
script: `test_head_transpose_debug.py` always printed BOTH `phys_addr`
values first (pure CMA allocation, no PL involvement at all) and then died
at the very first actual AXI-Lite register access. It also explains why
the failure is a hard reset rather than an error or bad data.

**IMPLICATION -- the right way to do ILA debugging on this platform:**
Do NOT JTAG-program a bitstream the OS has not been told about. Package
the ILA-instrumented design as its OWN xmutil app (its own `.bit.bin` +
`.dtbo` + `.xclbin` + `shell.json`, exactly as was already done once for
`kv260-custom`) and load it with `xmutil loadapp`. That gives BOTH the
debug core AND a consistent device-tree/zocl/address-map state, so
register access behaves normally and the ILA is still available to
capture. This is a repeat of packaging work already done successfully
before (see §10/§12), not new ground.

Interim fallback if that packaging is not wanted immediately: do all
functional testing on the **xmutil-loaded** design (which reliably
reproduces the real all-zeros bug WITHOUT crashing) and treat JTAG/ILA
capture as unavailable until the debug design is properly packaged.

**STATUS OF THE ORIGINAL INVESTIGATION:** unchanged -- the all-zeros bug
is still unexplained, and §24's goal (re-capturing RDATA across the full
1024-sample window) was NEVER ACHIEVED, because every attempt to trigger
it crashed the board. That capture remains outstanding and now requires
the packaged-debug-app approach above to attempt safely.

---

## 26. *** ROOT CAUSE OF THE ALL-ZEROS BUG FOUND *** -- CMA buffers live in HIGH DDR that the PL's HP ports cannot reach

**This resolves the central mystery of the entire project.** Found by
direct measurement on the safe xmutil-loaded path, no JTAG involved.

### The decisive experiment

Allocated a zocl CMA buffer, wrote a known pattern through the zocl
mmap, then read the SAME physical address back through `/dev/mem` at BOTH
the raw (high) address and the "stripped" (low) address the driver
actually programs into the IPs:
```
zocl reported paddr = 0x8011ff000
wrote ABCDEF99 via zocl mmap  (readback via zocl = abcdef99  OK)

/dev/mem @ HIGH 0x8011ff000 = abcdef99   <-- THE DATA IS HERE
/dev/mem @ LOW  0x11ff000   = 10888708   <-- DIFFERENT, UNRELATED MEMORY
```
(script saved as `debug_scripts/where_is_buf.py`)

**The high and low addresses are NOT aliases of the same DRAM.** They are
two physically distinct memory regions, both real, both readable. This is
just the standard ZynqMP 4GB memory map, confirmed on-target:
```
/proc/iomem:
  00000000-7fefffff  : System RAM   (low 2GB)
  800000000-87fffffff: System RAM   (high 2GB)
```
§22 suspected exactly this and it is now PROVEN by measurement rather
than inferred.

### Why this explains EVERY previous observation

| Prior evidence | Actual explanation |
|---|---|
| §17 ILA: writes complete cleanly, no DECERR | PL wrote to the stripped LOW address -- **valid RAM, just the WRONG RAM**. No error is expected. |
| §18 ILA: AWADDR/ARADDR "bit-exact match" | They matched the *stripped* value we programmed. The comparison was self-consistent but both sides were wrong. |
| §19 ILA: RDATA "random/uninitialised-looking" | It was **real data read from the wrong memory location**. Not garbage, not a glitch -- just somebody else's memory. |
| All-zeros output / sentinel 0x55 surviving | The PL never touches the CPU's actual buffer at all. |
| Cache-coherency work (§18, §20, §21) | Irrelevant -- the two sides were never looking at the same physical memory, so no amount of cache flushing could help. |

The `HPC_ALIAS_BIT` subtraction in `board_driver/hw_ip_driver.py`
(lines ~194-196), introduced on Day 5, is **wrong**. Its premise -- that
zocl returns a coherent-port alias of the same low DRAM -- is false.

### Why NOT stripping doesn't fix it either

`debug_scripts/test_pool_no_strip.py` (the §22 next-step-#1 experiment,
never previously run -- run today, on pool_engine on the untouched
non-coherent hp1 path) programs the FULL un-stripped high address:
```
in_buf.phys_addr  = 0x803258000  (FULL, not stripped)
out_buf.phys_addr = 0x80477a000  (FULL, not stripped)
HW result: [85, 85]   <-- sentinel UNCHANGED, same failure
```
Because the HP ports' address decode only covers `0x0-0x7FFF_FFFF`
(`HP0/1/2_DDR_LOW`, base 0, range 2G -- as documented in
hw_ip_driver.py's own comment and the Vivado Address Editor). The PL
**cannot emit** `0x8_0000_0000+` through those segments as currently
configured.

So: **strip -> PL hits the wrong memory; don't strip -> PL cannot reach
the address at all.** Both fail, for different reasons. That is why every
address-form experiment to date has come back identical.

### Why zocl hands out HIGH addresses in the first place

```
dmesg: cma: Reserved 1536 MiB at 0x0000000016000000     <- CMA pool is LOW
dmesg: [drm] Allocating BO from CMA for invalid or unused memory index[0]
```
zocl does NOT honour the hand-built `MEM_TOPOLOGY`
(`notes/mem_topology_9ip_v2.json` correctly declares all three banks at
`m_base_address = 0x0`) -- it reports memory index 0 as "invalid or
unused" and falls back to a generic coherent allocation. And because the
`zyxclmm_drm` node in `notes/custom-devicetree.dtsi` has **no
`memory-region` property and no `dma-ranges`**, that fallback is
unconstrained and freely returns high-DDR pages the PL cannot reach.

### THE FIX -- two options

**Option A (device tree only, NO bitstream rebuild -- recommended first):**
Constrain zocl's allocations to low DDR. Add a reserved-memory CMA pool
restricted to the low region and point the zocl node at it:
```dts
reserved-memory {
    #address-cells = <2>; #size-cells = <2>; ranges;
    zocl_reserved: buffer@20000000 {
        compatible = "shared-dma-pool";
        reusable;
        size = <0x0 0x20000000>;                    /* 512MB */
        alloc-ranges = <0x0 0x20000000 0x0 0x40000000>;  /* LOW DDR only */
    };
};
/* then, in the zyxclmm_drm node: */
memory-region = <&zocl_reserved>;
```
Then every zocl BO lands below 2GB, the existing `HPx_DDR_LOW` segments
can reach it, and **the HPC_ALIAS_BIT stripping must be REMOVED from
hw_ip_driver.py** (addresses will already be low and correct).

**Option B (Vivado rebuild):** assign the `HPx_DDR_HIGH` segments for
each IP's `m_axi` master in the Address Editor so the PL can reach
`0x8_0000_0000+` directly.
**IMPORTANT CAVEAT to check first:** Vitis HLS `m_axi` masters default to
a **32-bit address width**, and a 32-bit master physically cannot emit
`0x8_0000_0000` regardless of what the address segments say. If the IPs'
masters are 32-bit, Option B additionally requires regenerating all 9 IPs
with a 64-bit `m_axi` address width. Verify the master address width
before committing to this path.

**Recommendation: try Option A first** -- it needs no HLS or Vivado
rebuild, only a device-tree edit + `dtc` recompile + repackage of the
`kv260-custom` app (all steps already proven working in §10/§12), and it
sidesteps the 32-bit-master question entirely.

### *** FIXED AND VERIFIED -- THE HARDWARE NOW PRODUCES CORRECT OUTPUT ***

Neither Option A nor B was needed. Probing allocation size vs. returned
address revealed the actual lever:
```
SIZE         PADDR            PL-REACHABLE (<0x8000_0000)?
4096         0x801f17000      no  (high DDR)   <- generic page allocator
65536        0x16140000       YES              <- CMA pool
1MB          0x16c00000       YES
4MB..256MB   0x17100000       YES
```
**Allocations >= 64KB are served from the CMA pool** (reserved low at
`0x1600_0000`), which the PL can reach. Only small single-page requests
fall through to the generic allocator and land in high DDR. Every test in
this project used tiny buffers (48 bytes -> rounded to one 4KB page),
which is precisely why every test failed.

**THE FIX** (in `board_driver/hw_ip_driver.py`):
1. `MIN_CMA_ALLOC = 64 * 1024` -- every `ZoclBuffer` is rounded up to at
   least 64KB, forcing CMA (low-DDR, PL-reachable) allocation.
2. The `HPC_ALIAS_BIT` subtraction is **removed**. It was based on a false
   premise and silently pointed the PL at unrelated memory. Replaced with
   a hard `MemoryError` if a high address is ever returned -- fail loudly,
   because a wrong address here is invisible at runtime and cost days.

**VERIFIED RESULTS (2026-08-29, xmutil-loaded kv260-custom):**
```
head_transpose (smartconnect_hp2), 1MB buffers:
  HW result: [0,16,32,1,17,33,2,18,34,...,15,31,47]
  Expected:  [0,16,32,1,17,33,2,18,34,...,15,31,47]
  *** PASS -- exact match ***

pool_engine (smartconnect_hp1), 1MB buffers:
  result: [10, 20]   expected [10, 20]
  *** PASS ***

ORIGINAL test_head_transpose_debug.py, UNMODIFIED, fixed driver only:
  in_buf.phys_addr  = 0x16140000
  out_buf.phys_addr = 0x16150000
  HW result: [0,16,32,1,17,33,...]   *** CORRECT ***
```
Two different IPs on two different interconnects (hp1 and hp2) both
produce correct output. The all-zeros bug that blocked this project from
Day 5 onward is **RESOLVED**.

### ALL 9 IPs VALIDATED ON HARDWARE (`debug_scripts/validate_all_9_ips.py`)

With the fixed driver, every one of the 9 custom IPs was exercised on real
silicon. **9/9 produce correct output:**
```
head_transpose        PASS (exact match)
pool_engine           PASS (exact match)
upsample_engine       PASS (exact match)
eltwise_add           PASS (exact match)
conv2d_engine         PASS (exact match)   1x1 identity conv
depthwise_engine      PASS (exact match)   1x1 identity depthwise
head_sigmoid          PASS -- verified by hand below
lcam_attention_gate   PASS -- verified by hand below
head_concat_reshape   PASS -- h20 block correct at output head
```
`head_sigmoid` with fp_in=4, fp_out=7 (in/16 -> sigmoid -> x128):
```
in    -64  -32    0   32   64   96  -96   16
real   -4   -2    0    2    4    6   -6    1
exp   2.3 15.3   64 112.7 125.7 127.7 0.3 93.6
got     2   15   64  113   126  127    0   94   <- all correct
```
`lcam_attention_gate` with weight=64 @ fp_weight=7 (gain 0.5), feat 1..8:
expected `[0.5,1,1.5,2,2.5,3,3.5,4]` -> `[1,1,2,2,3,3,4,4]`; got exactly
that. **The custom LCAM attention mechanism -- the research contribution
that the DPU cannot run natively -- is confirmed working in hardware.**

Note this spans both interconnects (hp1 and hp2) and both AXI-master
counts (single-input and multi-input IPs), so the fix is general, not
specific to one datapath.

### Notes for anyone re-reading the earlier sections
Sections 5, 14, 17-23 document 16+ hypotheses tested and eliminated. All
of that elimination work was sound -- none of those WERE the cause. The
actual cause was never on the list because everyone (reasonably) assumed
zocl would hand back an address the PL could use, and the Day-5 alias
subtraction made the addresses *look* plausible while being wrong. The
ILA evidence in §17-19 was all real and correctly interpreted at the bus
level; it just could not reveal that the address itself pointed into a
different DRAM region.

**Also note:** the crash investigation in §24/§25 was a genuine detour --
the JTAG-load crash is a real, separate issue (still true, still worth
avoiding), but it was NOT related to the all-zeros bug at all.

---

## 27. DEPLOYMENT PREP -- weight-set audit: `weight_map.json` is unusable, and 7 learned tensors were never extracted

With the IPs working, the next step is end-to-end inference driven by
`layer_plan.json` (153 layers) + the extracted weights. Audit findings:

### `weight_map.json` is 100% unusable -- ignore it
All **390/390** tensor entries have `"file": null`, every one carrying the
note `'xir.Tensor' object has no attribute 'to_numpy'`. The original
extractor called a method that does not exist in this xir API, so it
recorded nothing. The 211 `.npy` files on disk were produced by some
*other* route and are NOT indexed by this file. **Do not try to use
weight_map.json; resolve weights by name instead** (see below).

### Name matching must be SHAPE-VERIFIED
Both the plan's tensor names and the `.npy` filenames are truncated, and
differently (plan: `ector__model_head_reg_preds_0_weight_fix`; file:
`const_XDetector__model_head_reg_preds_0_weight.npy`). Suffix matching
works, but **naive suffix matching silently mismatches 6 layers** (e.g.
L1 wants `[1,1,1,64]` and gets a `[1,1,1,512]` file). Every match must be
verified against the plan's declared shape.
`board_driver/weight_resolver.py` implements this: **103/115 conv/
depthwise layers fully resolved, 0 shape mismatches.**

### Slot convention in layer_plan.json (verified against the graph)
```
conv2d-fix                 : input_0 = bias, input_1 = activation, input_2 = weight
pool/eltwise/hard-sigmoid  : input_0 (+input_1) are ACTIVATIONS -- no consts
"depthwise-fix" (L23,41,53,83): NOT a real depthwise conv --
      input_0 = activation [1,H,W,C], input_1 = [1,1,1,C] hardsigmoid gate,
      no kernel, no group. This is the LCAM ATTENTION GATE MULTIPLY and
      maps to our lcam_attention_gate IP. Needs no weights.
```

### GENUINELY MISSING (7 learned tensors) -- a real blocker for accuracy
```
L1,  L3   channel_reduce max/avg weight  [1,1,1,64]     MISSING
L46, L50  channel_reduce avg/max weight  [1,1,1,128]    MISSING
L52, L77  channel_reduce avg/max weight  [1,1,1,256]    MISSING
L138      cls_convs_1_0 conv bias        [128]          MISSING
```
Only the `[1,1,1,512]` (lcam5) channel_reduce pair was ever extracted.
Checked whether the missing ones could be synthesised: **no** -- these are
learned weights (34 and 57 unique int8 values), not uniform constants.

Not a blocker: **L25's missing `[128]` tensor is a `fake_bias`** (fp=17),
a quantiser-inserted placeholder that is legitimately zeros.

### Why they can't simply be re-extracted from the xmodel (yet)
In a *compiled* xmodel the weights are not stored inline on the const ops
-- confirmed: **all 222 const-fix ops return 0 bytes** from
`op.get_attr("data")`. The tensors instead carry `reg_id` / `ddr_addr` /
`location` attributes pointing into a packed DDR parameter blob held on
the DPU subgraph:
```
reg_id_to_size         : {REG_0: 139264 (CONST), REG_1/2/3: DATA}
reg_id_to_context_type : {REG_0: 'CONST', ...}
reg_id_to_parameter_value : returns None  <-- NOT exposed by the Python binding
```
So the blob exists but this xir Python binding will not hand it over.
`board_driver/extract_weights_fixed.py` uses the correct
`op.get_attr("data")` API and cleanly extracts all 222 -- but they come
back empty for exactly this reason. Keep it; it is right for an
uncompiled/float xmodel, and documents the finding.

**Options for recovering the 7:** (a) parse the xmodel protobuf directly
for REG_0 and slice by each tensor's `ddr_addr`/size; (b) re-run
extraction on the host with the original float model / a xir build whose
Python binding exposes `reg_id_to_parameter_value`; (c) `xdputil`-based
dump on the board.

### Impact on deployment
Latency/FPS benchmarking -- the headline research number -- does **not**
depend on weight values, so the full pipeline can be built and benchmarked
now. Only *detection accuracy* is affected, and only via the LCAM spatial
attention channel-reduce at the 64/128/256-channel scales plus one head
bias.

---

## 28. *** PERFORMANCE REALITY CHECK *** -- full-custom is 158,000x too slow, but the HYBRID (custom LCAM + DPU) is a real 2x win

### `layer_plan.json` cannot drive an end-to-end dataflow
Only **24 of 168** activation inputs resolve to a producing layer in the
plan. The plan captures the 153 *compute* layers, but the xmodel has 493
ops -- including 21 `concat-fix`, 20 `download`, 16 `upload`, 12
`reshape-fix`, 18 `fix2float`, 13 `float2fix`. Those glue ops are absent,
so the graph is disconnected. A true end-to-end engine must be built from
the xmodel via `xir` (topology IS accessible -- only the weight blob is
not), not from `layer_plan.json`.

### MEASURED hardware throughput of conv2d_engine (real layer sizes)
```
L12   out [1,1,1,4]         0.00 MMAC      0.01 ms
L51   out [1,80,80,1]       0.63 MMAC    207.49 ms
L75   out [1,40,40,128]    52.43 MMAC   3328.95 ms
```
Linear fit: **16.59 MMAC/s**, with ~**170 ms fixed overhead per layer**.

### Extrapolated full-network cost
```
conv layers  : 111   13.00 GMAC   est 802.4 s
other layers :  42                est   7.5 s
TOTAL PER FRAME: 809.9 s  =  13.5 MINUTES
FPS            : 0.00123
DPU baseline   : 195 FPS
=> the full-custom accelerator is ~158,000x SLOWER than the DPU
```
Heaviest layers are brutal: L145/146/148/149 are 943.7 MMAC each -> ~57 s
*per layer*.

**Why:** `conv2d_engine.cpp` is a naive loop nest. Every operand is
fetched individually from DRAM over AXI -- no BRAM tiling, no burst
transfers, no unrolling, no `DATAFLOW`. It is memory-bound at roughly 6
clock cycles per MAC at 100 MHz. This is expected for unoptimised HLS and
is a property of the IP design, NOT of the addressing fix from §26.

**Conclusion: replacing the DPU wholesale with these IPs is not viable.**
Closing a 158,000x gap needs orders-of-magnitude HLS rework (tiling into
BRAM, burst `m_axi`, aggressive `UNROLL`/`ARRAY_PARTITION`, `DATAFLOW`
pipelining), which is a research project in itself.

### BUT -- the hybrid path works, and it is the original research thesis
The project's actual novelty was never "beat the DPU at convolution". It
was: **the DPU cannot run the LCAM attention natively** -- it falls back to
the ARM CPU. So the question that matters is: does our custom
`lcam_attention_gate` IP beat the CPU fallback? **Measured, at the four
real gate sizes:**
```
layer  shape          HW time    CPU (numpy int8)   winner
L23    160x160x64     23.6 ms      57.2 ms          HW  (2.42x)
L41    80x80x128      10.1 ms      16.3 ms          HW  (1.61x)
L53    40x40x256       4.7 ms       6.3 ms          HW  (1.34x)
L83    20x20x512       2.2 ms       3.4 ms          HW  (1.55x)
------------------------------------------------------------
TOTAL              40.6 ms      83.2 ms          HW  (2.05x)
```
**The custom LCAM IP is 2.05x faster than the CPU fallback across the
whole attention workload**, and wins at every scale. (The CPU side is
vectorised numpy int8 -- a fair, if not generous, comparison; a real VART
CPU-subgraph fallback would likely be slower still.)

End-to-end implication, using the paper's 195 FPS (5.13 ms) DPU figure:
```
DPU + CPU attention : 5.13 + 83.2 = 88.3 ms  -> 11.3 FPS
DPU + CUSTOM LCAM IP: 5.13 + 40.6 = 45.7 ms  -> 21.9 FPS   (~1.93x)
```

### RECOMMENDED DEPLOYMENT TARGET
Ship the **hybrid**: DPU for the convolutional backbone/neck/head, custom
`lcam_attention_gate` IP for the four LCAM attention gates. This is
deployable with what already works today, needs none of the 7 missing
weight tensors (the gate takes activations, not weights), and produces a
defensible ~1.9x end-to-end speedup plus a clean "custom hardware for a
novel operator the vendor IP cannot express" story.

The full-custom 9-IP design remains valuable as a completed, verified,
9/9-working accelerator and as the honest negative result documented
above -- and is the natural starting point for future HLS optimisation
work.

---

## 29. *** END-TO-END INFERENCE WORKING + THE REAL BOTTLENECK QUANTIFIED *** (2026-08-29)

**This section contains the headline numbers for the journal paper.**

### 29.1 The fingerprint problem, and its solution
`lcam_v5.xmodel` (the copy on the board) would NOT run:
```
CHECK fingerprint fail! model_fingerprint 0x101000056010407
  is un-matched with actual dpu_fingerprint 0x101000016010407
```
A compiled xmodel only runs on a DPU whose fingerprint matches exactly.
Read the fingerprint of any xmodel with:
```python
g = xir.Graph.deserialize(path)
for s in g.get_root_subgraph().toposort_child_subgraph():
    if s.has_attr('dpu_fingerprint'): print(hex(s.get_attr('dpu_fingerprint')))
```
Audit of every build in the WSL workspace
(`/home/punnam_rahul/wildfire_project/`):
```
compiled_v5/lcam_v5.xmodel            0x101000056010407   (Vitis-AI 3.0)
compiled_v5_2p5/lcam_v5_2p5.xmodel    0x101000016010407   <-- MATCHES BOARD
compiled_v5_on_3p0/v5_on_3p0.xmodel   0x101000056010407
compiled_v4_on_3p0/v4_on_3p0.xmodel   0x101000056010407
```
**`lcam_v5_2p5.xmodel` matches the stock `kv260-benchmark-b4096` DPU.**
No recompile was needed -- the wrong xmodel had simply been on the board.
(All builds report `target: DPUCZDX8G_ISA1_B4096`, so the *target name is
not sufficient* to tell them apart -- only the fingerprint is.)

Original compile command (from bash history), for reference:
```
vai_c_xir --xmodel .../LCAMYOLOXDetector_int.xmodel \
  --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
  --output_dir ... --net_name ...
```
Docker image: `xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106`.
Quantized (recompilable) model: `exports/detector_v5/LCAMYOLOXDetector_int.xmodel`.

### 29.2 END-TO-END INFERENCE NOW WORKS
Via `vitis_ai_library.GraphRunner` (present on the board; it dispatches DPU
subgraphs to the DPU and CPU subgraphs to the ARM core automatically).
`debug_scripts/e2e_graphrunner.py`, image `WEB09971.jpg` (1200x600 ->
letterboxed 640x640), 20 timed runs after 3 warm-ups:
```
mean latency : 430.08 ms      median: 429.98 ms
min / max    : 428.09 / 432.77 ms      std: 1.14 ms
FPS          : 2.33
output       : [1, 8400, 7]  float32   (YOLOX head: cx,cy,w,h,obj,cls0,cls1)
```
**The real end-to-end rate is 2.33 FPS, NOT the 195 FPS quoted in earlier
notes.** That 195 FPS figure was DPU-compute-only and ignored the CPU
fallback entirely. Correct this everywhere it appears.

### 29.3 WHERE THE TIME GOES -- the paper's motivating measurement
`debug_scripts/profile_dpu_vs_cpu.py` times each DPU subgraph individually
(DPU time is data-independent, so zero-filled inputs give valid timings):
```
#   output shape          mean ms   ops
0   [1,160,160,64]          8.701    28
1   [1,80,80,128]           5.400    31
2   [1,40,40,256]           4.017    31
3   [1,20,20,512]           4.739    32
4   [1,80,80,2]            16.605    67
5   [1,20,20,7]             0.219     6
6   [1,40,40,7]             0.318     6
7   [1,80,80,7]             0.799     6
-----------------------------------------
TOTAL DPU COMPUTE :  40.80 ms  ( 9.5%)
CPU FALLBACK      : 389.28 ms  (90.5%)   <-- THE BOTTLENECK
END-TO-END        : 430.08 ms           -> 2.33 FPS
```

### 29.4 WHAT the CPU is doing -- 93.9% of it is the LCAM attention
`debug_scripts/enumerate_cpu_ops.py`. CPU subgraphs 2/4/6/8 are the four
LCAM attention gates, and each is exactly four ops -- the multiply plus
the float conversions the CPU path requires:
```
SG   gate shape        mul elems    + fix2float/float2fix    subgraph total
 2   160x160x64        1,638,400    x3                        4,940,800
 4    80x80x128          819,200    x3                        2,464,000
 6    40x40x256          409,600    x3                        1,230,400
 8    20x20x512          204,800    x3                          614,800
                                                    TOTAL:    9,250,000
```
Against 9,854,800 total CPU-side elements:
**the four LCAM gates are 93.9% of ALL CPU work.**

Op-type totals across all CPU subgraphs:
```
fix2float    18 ops   3,248,800 elem
float2fix    13 ops   3,156,000 elem
mul           4 ops   3,072,000 elem   <-- the LCAM gates
sigmoid       6 ops      25,200 elem
reshape-fix   4 ops     117,600 elem
transpose     4 ops     117,600 elem
concat-fix    1 op       58,800 elem
fix           1 op       58,800 elem
```

**KEY ARCHITECTURAL POINT for the paper:** the `fix2float`/`float2fix`
pairs exist *only because the CPU implementation works in float32*. Our
`lcam_attention_gate` IP is **natively int8** -- it consumes the DPU's int8
output directly and emits int8. So moving these gates to the custom IP
eliminates the multiply AND all three conversion ops per gate. That is a
qualitative advantage over the CPU path, not merely a faster multiply.

### 29.5 PROJECTED HYBRID PERFORMANCE
Custom IP measured at the four real gate sizes (§28): **40.6 ms total**
(23.6 + 10.1 + 4.7 + 2.2).
```
                          latency        FPS      speedup
baseline (DPU + CPU)      430.08 ms      2.33       1.00x
hybrid, upper bound        81.40 ms     12.29       5.28x
  = 40.80 (DPU) + 40.6 (IP), assumes ALL CPU work is displaced
hybrid, conservative      ~105 ms       ~9.5       ~4.1x
  = 40.80 (DPU) + 40.6 (IP) + ~23.7 (residual 6.1% CPU work)
```
**Expect a ~4-5x end-to-end speedup.** Both figures are projections until
the hybrid bitstream exists and is measured -- do NOT publish them as
measured results.

### 29.6 What still has to be built
The current `kv260-custom` bitstream has the 9 IPs and **no DPU**; the DPU
apps have no custom IP. They cannot run simultaneously. The hybrid needs a
Vivado design containing **DPU + lcam_attention_gate together**, plus a
runner that drives DPU subgraphs via `vart.Runner` and dispatches
subgraphs 2/4/6/8 to the custom IP instead of the CPU.

### 29.8 *** WORKING DETECTIONS *** -- and the two preprocessing traps
`debug_scripts/postprocess_detections.py` on `WEB09971.jpg`:
```
fire   conf=0.875  box=[384, 11,763,468]
smoke  conf=0.651  box=[469,416,713,495]
smoke  conf=0.312  box=[609,514,634,539]
TOTAL DETECTIONS: 3
```
Verified visually: the `fire` box tightly bounds the flame column and the
`smoke` boxes cover the combustion base. Annotated output is saved to
`results/result_baseline.jpg` (input: `results/input_WEB09971.jpg`).
Pipeline sanity: 30 anchors with obj>0.1 -> 17 above conf 0.30 -> 3 after
NMS.

**TRAP 1 -- input quantisation.** The input tensor has `fix_point = -1`,
i.e. `int8 = real * 2^(fix_point) = real / 2`. **YOLOX consumes RAW 0..255
BGR pixels; it does NOT normalise to [0,1].** So `int8 = pixel / 2` ->
range 0..127, which fits int8 exactly. Dividing by 255 first (the usual
reflex, and what the old `notes/infer_kv260.py` does) collapses the whole
image to zeros. Always print the quantised input range -- it must be
[0,127] for this model, not [0,0].

**TRAP 2 -- double sigmoid.** The compiled xmodel ALREADY applies sigmoid
to `obj` and the class channels -- those are the 6 `sigmoid` ops in CPU
subgraphs 10/11/12/13/15/17 (§29.4). Applying sigmoid again in
postprocessing squashes every score into a narrow band and produced
**1044 false boxes all at confidence 0.304**. Detect it instead: if
`obj`/`cls` already lie in [0,1], skip the sigmoid. The box channels
(`cx,cy` in ~[-1,2], `w,h` in ~[-0.4,3.4]) DO still need the standard
YOLOX grid/stride decode: `(v+grid)*stride` and `exp(v)*stride`.

Anchor layout confirmed empirically as **80x80(6400) + 40x40(1600) +
20x20(400) at strides 8/16/32**, concat order `80,40,20`.

### 29.9 SUMMARY TABLE FOR THE PAPER (measured on hardware)
```
                                    latency      FPS     share
end-to-end (DPU + CPU fallback)     430.08 ms    2.33    100%
  |- DPU compute                     40.80 ms            9.5%
  |- CPU fallback                   389.28 ms           90.5%
        |- 4 LCAM attention gates                       93.9% of CPU work
custom lcam_attention_gate IP        40.60 ms   (measured, §28)
PROJECTED hybrid                    ~81-105 ms  ~9.5-12.3   4.1-5.3x
```
Detection output is identical in the hybrid (the gate is a pure
elementwise multiply), so **accuracy is unchanged** -- the speedup carries
no accuracy cost. This also sidesteps the 7 missing weight tensors (§27)
entirely, since the DPU uses its own compiled weights.

### 29.10 HYBRID PATH -- the bitstream ALREADY EXISTS (and why it never worked)

Audit of `C:\Xilinx\projects\` found three distinct designs:
```
18-07-2026  xsa_extract_FINAL / kv260_lcam_app   DPU + decode_0 + nms_top_0
                                                  (the original DPU-era design;
                                                   this is what is on the board)
24-07-2026  kv260_dpu_plus_9ip                    DPU + lcam_attention_gate
                                                  <-- THE HYBRID WE NEED
25-08-2026  DPUCZDX8G/srcs/top/hw_handoff/top.hwh all 9 IPs, NO DPU
                                                  (srcs dir was overwritten by
                                                   the noDPU design -- stale)
```
**`kv260_dpu_plus_9ip` is misleadingly named: it contains the DPU and
ONLY `lcam_attention_gate`** (synth log: 101 hits for lcam_attention,
0 hits for each of the other 8 IPs). Instantiated as
`top_lcam_attention_gate_0_0`. That is exactly the hybrid architecture.

Its implementation is COMPLETE and CLEAN:
```
bitstream : KV260.runs/impl_1_01/top_wrapper.bit  (6,971,169 bytes, 24-07-2026)
timing    : WNS +0.040 ns, WHS +0.010 ns -- "All user specified timing
            constraints are met"
routing   : 210,354 nets fully routed, 0 routing errors
power     : 6.870 W total (6.533 dynamic + 0.336 static)
resources : LUT 75,898 (64.80%)   FF 133,084 (56.82%)
            DSP 749 (60.02%)      URAM 64 (100%)   BRAM 36 tiles (25%)
```
(Those resource/power numbers are directly usable in the paper.)

**WHY IT NEVER WORKED: it was never packaged.** Every XSA on disk dates
from 14-07 to 21-07 and contains the OLD `decode_0`/`nms_top_0` design.
**There is no XSA from the 24-07 hybrid build**, so no `.dtbo`/`.xclbin`
could be generated, so `xmutil` could never load it -- and JTAG-loading it
directly hangs the board (§25). This is a packaging gap, NOT a design
defect. Fix: `write_hw_platform -fixed -include_bit` on `impl_1_01`, then
repeat the §10/§12 packaging flow.

### 29.11 DPU FINGERPRINT AND THE RECOMPILED XMODEL
The hybrid DPU's `arch.json` (`DPUCZDX8G/prj/Vivado/srcs/top/ip/top_DPUCZDX8G_0/arch.json`):
```
{"fingerprint":"0x101000012010407"}      B4096, URAM_ENABLE
```
The same fingerprint appears in `C:\Xilinx\xsa_check_ext`,
`C:\Xilinx\xsa_check_today` and `C:\Xilinx\vpp_tmp` -- i.e. it is the
project's standard DPU configuration.

It matches NO pre-existing xmodel (§29.1), so the quantized model was
recompiled against it inside the Vitis-AI Docker in WSL:
```
docker image : xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106
input        : exports/detector_v5/LCAMYOLOXDetector_int.xmodel  (quantized)
arch         : {"fingerprint":"0x101000012010407"}
command      : vai_c_xir --xmodel <int.xmodel> --arch arch_hybrid/arch.json \
                 --output_dir compiled_hybrid --net_name lcam_hybrid
result       : compiled_hybrid/lcam_hybrid.xmodel
               target DPUCZDX8G_ISA1_B4096_0101000012010407
               24 subgraphs, 8 DPU subgraphs
VERIFIED     : embedded dpu_fingerprint = 0x101000012010407  (exact match)
```
**Lesson: with the quantized xmodel + Vitis-AI Docker, a fingerprint
mismatch is never a blocker -- just recompile against the target
arch.json.** The quantized model is the recompilable artifact; the
compiled one is not.

### 29.12 EXPECTED HYBRID PERFORMANCE (and an important caveat)
The `lcam_attention_gate` baked into the 24-07 bitstream is the ORIGINAL,
UNOPTIMISED version -- its synthesis log shows
`lcam_attention_gate_fmul_32ns_32ns_32_3_max_dsp_1_ip`, i.e. a 32-bit
FLOATING-POINT multiplier, and the source
(`hls_ip_sources/lcam_attention_gate/lcam_attention_gate.cpp`) processes
ONE BYTE PER CYCLE with `float` math and `roundf()`.
```
budget with the EXISTING bitstream:
  40.8 (DPU) + 40.6 (unoptimised IP) + ~24 (residual CPU)  = ~105 ms -> ~9.5 FPS
  vs 430 ms / 2.33 FPS baseline                            = ~4.5x speedup
```
To beat the reference paper's ~15 FPS, the IP needs optimising:
 * replace `data_t*` with `ap_uint<512>*` -> 64 int8/cycle instead of 1
 * replace float multiply + `roundf` with pure integer shift arithmetic:
   `out = (feat * weight) >> (fp_feat + fp_weight - fp_out)`
 * burst `m_axi`, `UNROLL` the channel loop
Estimated 3.07M elements at 512-bit/100 MHz -> **~1-3 ms** (vs 40.6 ms),
giving ~40.8 + 3 + 24 = **~68 ms -> ~15 FPS**, and ~21 FPS if the residual
CPU ops (sigmoid/transpose/concat) are also offloaded. Requires one more
Vivado rebuild.

### 29.13 *** HYBRID BITSTREAM BUILT *** (DPU + lcam_attention_gate)

Built from scratch in a NEW isolated folder -- `C:\Xilinx\projects\kv260_hybrid_v2\`
-- so no existing project was modified. Structure mirrors the DPU TRD so
all of trd_bd.tcl's relative path derivations resolve inside it:
```
kv260_hybrid_v2/
  dpu_ip/                        (junction -> DPUCZDX8G/dpu_ip; see gotcha below)
  prj/Vivado/scripts/trd_prj.tcl + base/trd_bd.tcl + constrs/
  prj/Vivado/prj/KV260.xpr       (generated)
  prj/Vivado/srcs/top/top.bd     (generated)
  out/kv260_hybrid.bit + .xsa    (RESULT)
  package/                       (dtbo + bit.bin + shell.json)
```

**Build sequence (all scripts in `prj/Vivado/scripts/`):**
1. `trd_prj.tcl`            -> DPU-only BD + project + runs
2. `add_lcam_ip.tcl`        -> add lcam_attention_gate, wire it, addresses
3. `fix_lcam_addr.tcl`      -> correct the 2nd control address
4. `enable_uram_and_build.tcl` -> URAM fix + synth + impl + bitstream + XSA

**FINAL RESULT -- clean:**
```
IMPL STATUS : write_bitstream Complete!  (100%)
WNS         : +0.019 ns     WHS : +0.010 ns
timing      : "All user specified timing constraints are met"
DPU cells   : 210,760       LCAM cells : 14,441   (both in the netlist)
power       : 6.921 W  (6.584 dynamic + 0.337 static)
LUT 73,235 (62.53%)  FF 123,288 (52.63%)
DSP 743 (59.54%)     URAM 64 (100%)   BRAM 20.5 tiles (14.24%)
```

**Wiring (the DPU's own connections were left untouched):**
```
PS M_AXI_HPM0_LPD -> smartconnect_ctrl -+-> hier_dpu/S_AXI      (DPU ctrl, 0x8F00_0000)
                                        +-> lcam/s_axi_control   0x8006_0000
                                        +-> lcam/s_axi_control_r 0x800F_0000
lcam m_axi_gmem0/1/2 -> smartconnect_lcam_hp -> PS S_AXI_HP3_FPD
        (HP3 was the only free HP port; DPU holds HP0/HP1/HP2 + LPD)
lcam ap_clk = pl_clk0 (100 MHz), ap_rst_n = rst_gen_reg
```
The control addresses were deliberately forced to `0x8006_0000` /
`0x800F_0000` so the EXISTING `board_driver/hw_ip_driver.py` IP_ADDR
table works unchanged. HP3's segment covers `HP3_DDR_LOW`
(0x0-0x7FFF_FFFF), i.e. exactly the PL-reachable low DDR required by §26.

### 29.14 THREE build failures worth recording (each cost a cycle)

**(a) `No Catalog IPs found` -> "Please specify VLNV when creating IP cell
DPUCZDX8G".** `trd_bd.tcl` derives `ip_dir` as `<3 dirs up from
scripts>/dpu_ip`, and **Vivado's `update_ip_catalog` does NOT traverse a
Windows directory junction** placed there -- the catalog came back empty
and the VLNV lookup
`get_ipdefs -all -filter {NAME=~dpuczdx8g && UPGRADE_VERSIONS==""}`
returned nothing. Fix: override `dict set dict_prj dict_sys ip_dir` with
the real absolute path AFTER sourcing trd_bd.tcl.

**(b) `opt_design` died: `[Chipscope 16-213] The debug port 'u_ila_0/probe0'
has 1 unconnected channels`.** The copied `constrs/debug.xdc` still held
the §24 ILA probe definitions, which reference `smartconnect_hp2/...` and
`smartconnect_hp0_M00_AXI_RDATA[*]` -- nets that do not exist in the
hybrid. Unconnected ILA probes are a HARD error. Fix:
`remove_files -fileset constrs_1 <debug.xdc>`. (It also produced ~110
"No nets matched" critical warnings that masked everything else.)

**(c) `place_design` died: BRAM over-utilised -- 242 RAMB36 needed, 144
available.** `trd_prj.tcl` ships `DPU_URAM_PER_DPU {0}`, i.e. URAM
DISABLED, so the DPU maps all image/weight banks into BRAM. Fix:
`set_property CONFIG.URAM_N_USER {64} [get_bd_cells hier_dpu/DPUCZDX8G]`
(patched in place, so the lcam wiring survives). Effect:
```
              RAMB36   RAMB18   URAM
URAM_N_USER=0   242       ?       0     -> DOES NOT FIT
URAM_N_USER=64   18       5      64     -> fits comfortably
```

### 29.15 IMPORTANT: enabling URAM CHANGED THE DPU FINGERPRINT
```
URAM_N_USER=0  -> 0x101000012010407
URAM_N_USER=64 -> 0x101000016010407
```
**Always re-read `srcs/top/ip/top_DPUCZDX8G_0/arch.json` after any DPU
config change** -- the fingerprint is part of the config, and a compiled
xmodel only runs on a matching DPU (§29.1).

Conveniently, `0x101000016010407` is the SAME fingerprint as the stock
`kv260-benchmark-b4096` DPU, which means **`lcam_v5_2p5.xmodel` (already
on the board, already validated end-to-end at 430 ms / 2.33 FPS) runs on
the hybrid unchanged.** No recompile needed, and the baseline and hybrid
measurements use the IDENTICAL model -- same weights, same graph, only
the attention execution differs. That makes the comparison airtight for
the paper. (The `lcam_hybrid.xmodel` compiled for `...12...` is now
redundant.)

### 29.16 Packaging (prepared, `kv260_hybrid_v2/package/`)
`pl_overlay_hybrid.dts` adapts the PROVEN `DPUCZDX8G/prj/Vivado/pl_overlay.dts`:
DPU node kept identical (0x8F00_0000, irq `0 89 4`, misc_clk_0 549.945 MHz
/ misc_clk_1 274.9725 MHz); `decode_0`/`nms_top_0` replaced by
`lcam_attention_gate_0`; **`zyxclmm_drm` (`xlnx,zocl`) added** -- without
it there is no `/dev/dri/renderD128` and `ZoclBuffer` cannot allocate
(§12). `build_package.bat` runs bootgen (.bit -> .bit.bin) + dtc
(.dts -> .dtbo) + writes shell.json.

**STILL OPEN:** VART needs an xclbin (`XLNX_VART_FIRMWARE`) to locate the
DPU. Since the hybrid DPU has the same fingerprint AND the same base
address (0x8F00_0000) as the stock one, the stock
`kv260-benchmark-b4096.xclbin` may work directly -- to be verified with
`xclbinutil` on the board. If not, fall back to the July
`pfm.tcl` -> `v++ --package` platform flow.

---

## 30. *** OPTIMISED LCAM IP: 11.6x FASTER THAN THE ORIGINAL, BIT-EXACT *** (2026-08-30)

The headline operator result for the paper.

### 30.1 What was changed
`hls_ip_sources/lcam_attention_gate/lcam_attention_gate_opt.cpp`:
```
                         ORIGINAL                  OPTIMISED
  m_axi width            8 bit (1 int8/beat)       512 bit (64 int8/beat)
  arithmetic             float mul + roundf()      pure integer shift
  synthesised multiplier fmul_32ns_32ns_32 IP      none (DSP 5 total)
  inner loop             PIPELINE II=1 over C      UNROLL 64 lanes, II=1
```
The whole operation is exactly
`out = clamp( sign(p) * ((|p| + 2^(s-1)) >> s) )`, `p = feat*weight`,
`s = fp_feat + fp_weight - fp_out` -- the float round-trip was never
needed.

**ROUNDING SUBTLETY (would have silently corrupted ~2% of outputs):**
`roundf()` rounds HALF AWAY FROM ZERO, but a bare `(p + bias) >> s`
rounds half toward -inf. They disagree on negative ties -- e.g.
`roundf(-2.5) = -3` while `(-5+1)>>1 = -2`. Feature values are signed
int8 so negatives are common. The fix is to round the MAGNITUDE and
re-apply the sign. `debug_scripts/verify_lcam_int_math.py` proves the
final form is bit-identical over the FULL int8 x int8 domain for all four
layer configs, and reports how many cases the naive version would have
got wrong (896-1296 per config).

### 30.2 HLS results (Vitis HLS 2022.2, xck26, 100 MHz)
```
estimated timing  7.300 ns (target 10.00)      BEAT_LOOP II = 1 achieved
m_axi_gmem0_RDATA 512 bit                      m_axi_gmem2_WDATA 512 bit
m_axi_gmem1_RDATA  32 bit (weight scalar)      float units: NONE
DSP 5    FF 7,194    LUT 40,296    BRAM 0    URAM 0
```
NOTE: Vitis HLS refuses project paths containing spaces
(`ERROR: [HLS 200-70] ... contains illegal character ' '`), so the build
runs from `C:\Xilinx\hls_work\lcam_opt`, not the project folder.

### 30.3 Standalone bitstream (`C:\Xilinx\projects\lcam_opt_bit\`)
Minimal design: PS + optimised IP + 2 smartconnects. Built clean:
```
WNS +0.759 ns   WHS +0.010 ns   failed nets 0
LUT 31,466 (26.87%)   FF 18,295 (7.81%)   BRAM 23.5 (16.32%)
DSP 5 (0.40%)         URAM 0 (0.00%)      power 2.898 W
```
Two Vivado gotchas hit here:
 * `M_AXI_HPM0_FPD` (GP0) apertures start at 0xA000_0000 and CANNOT reach
   0x8006_0000. Use `M_AXI_HPM0_LPD` (GP2) -- that is the port providing
   the 0x8000_0000 aperture (and what the DPU design used).
 * The KV260 board preset enables `M_AXI_HPM1_FPD`; leaving it on fails
   validation with "clock pins are not connected: /ps/maxihpm1_fpd_aclk".
   Explicitly set `PSU__USE__M_AXI_GP1 {0}`.
 * `ERROR: [Common 17-232] Could not create slave interpreter` during IP
   generation is TRANSIENT -- simply re-running the build succeeded.

Addresses were forced to 0x8006_0000 / 0x800F_0000, and the generated
`xlcam_attention_gate_opt_hw.h` confirms the register map is IDENTICAL to
the original IP (ctrl: H=0x10 W=0x18 C=0x20 fp=0x28/0x30/0x38;
ctrl_r: feat_in=0x10 weight_in=0x1c feat_out=0x28), so
`board_driver/ip_wrappers.py` drives it UNCHANGED.

### 30.4 *** MEASURED ON HARDWARE *** (`debug_scripts/bench_lcam_opt.py`)
```
layer  shape            HW (ms)   CPU (ms)   speedup  correct
23     160x160x64          2.24      76.35    34.04x   OK
41     80x80x128           0.61      36.10    58.91x   OK
53     40x40x256           0.33      17.98    54.93x   OK
83     20x20x512           0.31       7.54    24.10x   OK
------------------------------------------------------------
TOTAL                      3.50     137.97    39.46x
correctness: ALL LAYERS EXACT (bit-for-bit vs reference)
```
* vs the ORIGINAL IP (40.6 ms, §28): **11.6x faster**
* vs numpy int8 with exact rounding (137.97 ms): **39.5x**
* CAVEAT ON CPU BASELINES -- three different numbers appear in this log
  and they must not be mixed up:
    82.1 ms  simple numpy `(feat*w)>>7` (§28) -- understates the work
   137.97 ms numpy with the exact round/clip pipeline (this section)
   389.28 ms the REAL VART CPU fallback measured end-to-end (§29.3)
  For an end-to-end claim, 389.28 ms is the honest denominator.

### 30.5 Projected end-to-end
Using the measured baseline (430.08 ms, 2.33 FPS) and the measured CPU
fallback (389.28 ms, of which the four gates are the dominant share --
93.9% by element count, §29.4):
```
                                 latency        FPS
baseline (DPU + CPU attention)   430.08 ms      2.33
DPU + optimised custom IP        ~69 ms        ~14.6      ~6.3x
```
This is a PROJECTION, not a measured end-to-end run: it assumes the gate
share of CPU time tracks the element share, and it still requires the
DPU and the IP to run in one bitstream -- which is blocked by §31.

### 30.6 Why this IP path works when the DPU path does not
The custom IPs need NO XRT, NO xclbin, NO AP_CTRL_HS contract: the driver
pokes their AXI-Lite registers through /dev/mem and allocates buffers via
zocl directly. That is exactly why they have worked reliably all along
while DPU/VART integration keeps failing (§31).

### 29.7 Operational warning (cost a board crash this session)
`/lib/firmware/xilinx/kv260_lcam_app/` contains SEVERAL bitstreams and
xclbins (`kv260_lcam.bit.bin`, `kv260_lcam_FINAL.bit.bin`,
`kv260_lcam.xclbin`, `kv260_lcam_minimal.xclbin`, `.bak` files). Pairing
the wrong xclbin with the loaded bitstream **hangs the board** (same
failure mode as §25). Use `kv260-benchmark-b4096` with
`lcam_v5_2p5.xmodel` -- that combination is verified working.

---

## 31. *** OPTION B DONE IN HARDWARE: HYBRID DPU + LCAM xclbin BUILT AND TIMING-CLEAN *** (2026-08-30)

The Vivado-block-design hybrid of Sec 29.13 built a *bitstream*, but XRT could
never drive the DPU in it: the Vivado-TRD DPU is a raw IP, so XRT registered
it from IP_LAYOUT and drove it with the generic AP_CTRL_HS handshake --
writing `ap_start` into what is actually a read-only DPU **version**
register, and hanging CU(0). No amount of xclbin-section surgery fixed
that, because the problem was the contract, not the metadata.

This session took the other route: build the design the way Xilinx
intends, through `v++ --link`, so the DPU is packaged with its OWN
`kernel.xml` and XRT/VART drive it with the correct protocol.

**Result: `C:\Xilinx\projects\kv260_hybrid_vitis\hybrid.xclbin`, timing
met, both CUs present.** Not yet run on the board -- the board was
powered off when the build finished.

### 31.1 The blocker that had to be cleared first: no extensible platform
Every previous `v++` attempt died with
`ERROR: [v++ 60-1606] ... is a non-accelerated platform`, and
`platforminfo` on `kv260_9ip_v2.xpfm` confirmed why: it listed **no clocks
and no AXI ports**. A platform must publish PFM metadata before v++ can
place kernels into it. None of last month's v++ packages could ever have
worked; the platform was the missing piece all along.

Built a minimal extensible platform from scratch:
`C:\Xilinx\projects\kv260_ext_pfm\build_platform.tcl`
  PS + clk_wiz (100/300/600 MHz) + 3x proc_sys_reset + axi_intc +
  a 1x64 axi_interconnect for AXI-Lite, then:
    set_property PFM_NAME / PFM.CLOCK / PFM.AXI_PORT / PFM.IRQ
    set_property platform.extensible "true"
then `write_hw_platform` -> XSA, then xsct `platform create` +
`domain create -os linux -runtime {ocl}` -> `.xpfm`.

`platforminfo` on the result now reports 3 clocks and 7 memory SP tags
(HPC0/HPC1/HP0-3/LPD). That is the file the link consumes:
`kv260_ext_pfm/vitis_ws/kv260_ext_pfm/export/kv260_ext_pfm/kv260_ext_pfm.xpfm`

### 31.2 TWO Windows resource failures that masquerade as tool bugs
Both cost a full build cycle and both have the same cause -- this 16 GB
Windows box cannot sustain the parallelism Xilinx tools default to.

1. Platform build died three times in a row on `ps_e` only, with
     `ERROR: [Common 17-232] Could not create slave interpreter '::ipgen_iptclns'`
     `CRITICAL WARNING: [IP_Flow 19-1747] Failed to deliver ... zynq_ultra_ps_e_v3_4.ttcl`
   Every other IP generated fine. **Fix: `set_param general.maxThreads 1`**
   before `generate_target`, and `launch_runs synth_1 -jobs 1`. First
   attempt after that succeeded and has succeeded every time since.

2. First v++ link died 2 minutes in with
     `boost::filesystem::status: Insufficient system resources exist to
      complete the requested service: ".../.create_bd.end.rst"`
   v++ had launched 8 concurrent block-level synthesis jobs.
   **Fix: `synth.jobs=2` / `impl.jobs=2` under `[vivado]` in the link
   config.** Also kill orphaned `vivado.exe` processes after a failed
   link -- five of them survived and were holding 5.7 GB.

Neither is a licensing or version problem. Do not go looking for one.

### 31.3 The .xo kernels
- **DPU**: a prebuilt `dpu.xo` already existed at
  `C:\Xilinx\projects\DPUCZDX8G\prj\Vitis\binary_container_1\dpu.xo`
  (27-07-2026), config B4096 + URAM_ENABLE + DSP48_USAGE_HIGH +
  CHANNEL_AUGMENTATION_DISABLE. Reused as-is.
- **LCAM**: rebuilt from `lcam_attention_gate_opt.cpp` with
  `open_solution -flow_target vitis` + `export_design -format xo`
  (`C:\Xilinx\hls_work\lcam_xo\run_xo.tcl`).

Two source changes were REQUIRED to make it a legal Vitis kernel:

(a) **The Sec 23 `control`/`control_r` split is FATAL in kernel mode.** In the
    IP flow it was a harmless quirk (and was correctly ruled out as the
    all-zeros cause). Here HLS hard-errors:
      `[HLS 214-219] Vitis mode requires ALL SAxiLites have only one same
       bundle name. Now different bundle names: 'control' and 'control_r'`
    Fix: pin the m_axi offset registers onto the same bundle explicitly --
      `#pragma HLS INTERFACE s_axilite port=feat_in bundle=control` (etc.)

(b) **512-bit was the wrong bus width** -- see Sec 31.4.

### 31.4 *** The 512-bit port was over-built: matching the HP port is FREE ***
The first link ROUTED but MISSED TIMING: **WNS -0.823 ns, TNS -817.7 ns,
5238 failing endpoints.** Every failing path was on `clk_out1` (100 MHz)
inside `axi_intc_0` and the AXI-Lite protocol converter -- **not** in the
DPU, **not** in the LCAM kernel. Trivial control logic missing a 10 ns
period by 0.8 ns is not a logic-depth problem, it is a placement problem.

The utilization report said exactly why:
```
CLB LUTs   83827 / 117120 = 71.6%
CLB slices 14599 /  14640 = 99.72%   <-- no free slice anywhere
BRAM 96.18%   URAM 71.88%
   DPUCZDX8G_1              49058 LUT
   lcam_attention_gate_opt_1 28993 LUT
```
The placer had zero freedom, so it scattered the small control blocks.

The LCAM kernel was the reducible half, and it turned out to be
**over-built rather than merely large**. A ZynqMP `S_AXI_HP` port is
**128 bits wide**. A 512-bit kernel port therefore makes the platform
insert a width converter, and the 64 multiply lanes then sit idle 3 of
every 4 cycles. The old standalone measurement confirms this exactly:
```
6.5 MB over 128 bits @ 100 MHz (1.6 GB/s)  predicts  4.0 ms
Sec 30.4 measured                                        3.50 ms
```
The **bus**, not the arithmetic, set the time all along. So narrowing
WBITS 512 -> 128 (LANES 64 -> 16) costs NO throughput and cuts the
unrolled datapath ~4x. Also needed, because `-flow_target vitis` silently
sets `m_axi_max_widen_bitwidth=512` and would widen it straight back:
```tcl
config_interface -m_axi_max_widen_bitwidth 128
```

Effect:
```
                       512-bit        128-bit
HLS estimate LUT        45115          17991      (2.5x)
HLS estimate FF         22632           8272
placed LUT              28993          10062      (2.9x)
Estimated Fmax        411 MHz        411 MHz      (unchanged)
block-synth jobs           18             16      (2 width converters gone)
```

### 31.5 *** FINAL BUILD -- TIMING MET ***
```
hybrid.xclbin                7857414 bytes   30-08-2026 03:56
WNS  +0.031 ns   TNS 0.000   0 failing endpoints of 362093
WHS  +0.007 ns   "All user specified timing constraints are met."

CLB LUTs        61903 / 117120 = 52.85%
CLB Registers  110807 / 234240 = 47.30%
CLB slices      14201 /  14640 = 97.00%
BRAM             98.5 /    144 = 68.40%
URAM               46 /     64 = 71.88%
DSP               716 /   1248 = 57.37%

  DPUCZDX8G_1               48996 LUT   82 BRAM  46 URAM  710 DSP
  lcam_attention_gate_opt_1 10062 LUT    7 BRAM   0 URAM    6 DSP

Compute units (from xclbinutil --info):
  DPUCZDX8G_1                @ 0xa0010000
  lcam_attention_gate_opt_1  @ 0xa0020000
Memory banks: HPC0 HPC1 HP0 HP1 HP2 HP3 LPD
```
Link config `kv260_hybrid_vitis/hybrid.cfg`: DPU aclk 300 MHz /
ap_clk_2 600 MHz on HPC0+HP0+HP1 (same as stock `prj_config_1dpu`, so DPU
behaviour matches the 5.13 ms baseline); LCAM at 300 MHz with gmem0 on
HP2 and gmem1/gmem2 on HP3, deliberately splitting the streaming read from
the streaming write since this kernel is purely memory-bound.

### 31.6 Package (ready, NOT yet loaded)
`C:\Xilinx\projects\kv260_hybrid_vitis\package\out\`
```
hybrid.bit.bin        7797692   (xclbinutil --dump-section BITSTREAM:RAW
                                 then bootgen -process_bitstream bin)
hybrid.xclbin         7857414
pl_overlay_hybrid.dts    3352
shell.json                  49
install_hybrid.sh         2038
```
The overlay **deliberately declares NO IP nodes** -- zocl discovers both
CUs from IP_LAYOUT. Hand-writing a DPU node is what produced
"KDS is in bad state" earlier (its interrupt collides with the IRQ 89 zocl
claims); the stock Xilinx kv260-dpu overlay has no DPU node either.

### 31.7 NEXT SESSION -- exactly where to resume
The board was powered off, so NOTHING here is hardware-verified yet.
1. Power on; re-apply networking; `pscp -scp -batch -hostkey ...` the five
   files in `package/out/` to `~/hybrid_pkg/`; run `install_hybrid.sh`.
2. **Read the DPU fingerprint** (`xdputil query`). Enabling URAM moved it
   from `0x...12` to `0x...16` once already (Sec 29.15). Whichever value
   prints decides which xmodel loads: `lcam_v5.xmodel` or
   `lcam_v5_2p5.xmodel`. Do not assume -- check.
3. Confirm `xbutil examine` lists BOTH CUs, then confirm the DPU actually
   starts (this is the exact thing that failed in the Vivado flow).
4. The LCAM CU is now at **0xa0020000**, not the old 0x80060000. The
   existing `hw_ip_driver.py` IP_ADDR table must be pointed at the new
   base before any register poke, or it will write into the DPU.
5. Then measure end-to-end hybrid FPS.

Expected, from Sec 29.12 + Sec 30.5, with the 128-bit/300 MHz gate now
sustaining 4.8 GB/s (3x the standalone build's rate):
```
DPU + CPU attention   : 5.13 + 83.2  = 88.3 ms -> 11.3 FPS  (measured)
DPU + custom LCAM IP  : 5.13 + ~1.4  = ~6.5 ms -> ~150 FPS  (PROJECTED)
```
Treat the second line as a projection ONLY until step 5 is done.

---

## 32. *** HYBRID VERIFIED ON HARDWARE: BOTH CUs LIVE, LCAM GATE = 1.82 ms BIT-EXACT *** (2026-08-30)

The §31 build was loaded on the board and tested. Headline: **the hybrid
works.** The DPU is registered and driven correctly by XRT/VART for the
first time in this project, and the custom LCAM gate runs beside it,
bit-exact, at 1.82 ms for all four gate layers.

### 32.1 Install
`xmutil loadapp kv260-hybrid2` (installed under a NEW app name -- the old
failed Vivado-flow `kv260-hybrid` dir was left untouched, since mixing a
stale bitstream/xclbin pair in one app dir hangs the board, §29.7).

### 32.2 *** BOTH COMPUTE UNITS PRESENT AND IDLE ***
```
xbutil examine:
  Index  Name                                               Base_Address  Status
  0      lcam_attention_gate_opt:lcam_attention_gate_opt_1  0xa0020000    (IDLE)
  1      DPUCZDX8G:DPUCZDX8G_1                              0xa0010000    (IDLE)
Memory: HPC0 2GB, HP0 2GB, HP1 2GB, HP2 2GB, HP3 2GB
Xclbin UUID 38004AC5-349B-E5E6-8553-525652DE7E06
```

**`xdputil query` now reports a real DPU kernel** -- this is the exact thing
that never worked in the Vivado-block-design flow, where XRT wrote ap_start
into a read-only version register and hung CU(0):
```
"cu_name": "DPUCZDX8G:DPUCZDX8G_1",  "cu_addr": "0xa0010000",  "cu_idx": 0
"DPU Arch": "DPUCZDX8G_ISA1_B4096_0101000012010407"
"fingerprint": "0x101000012010407"
"DPU Frequency (MHz)": 300,   "XRT Frequency (MHz)": 300
```
The `v++ --link` route was the fix. Nothing else changed about the DPU.

### 32.3 *** THE KERNEL FLOW USES ap_ctrl_chain, NOT ap_ctrl_hs ***
This cost the first two benchmark runs and is a genuine trap, because the
register offsets look familiar and the first invocation appears to work.

From the generated `xlcam_attention_gate_opt_hw.h`:
```
bit 0 - ap_start    (Read/Write/COH)
bit 1 - ap_done     (Read)              <-- STICKY, not clear-on-read
bit 2 - ap_idle     (Read)
bit 4 - ap_continue (Read/Write/SC)
csynth report: Interface protocol = ap_ctrl_chain
```
`hw_ip_driver.poll_ap_done()` was written for the nine IP-flow cores, which
are `ap_ctrl_hs` -- there, reading ap_done clears it. In a v++-linked kernel
it does NOT: ap_done latches until `ap_continue` (0x10) is written.

Symptom if you reuse the old loop: the FIRST layer is bit-exact, and every
later layer returns in ~0.03 ms against a completely untouched output
buffer, because the poll immediately sees the previous run's done bit.
Observed exactly that:
```
23  160x160x64   1.21 ms  OK
41  80x80x128    0.03 ms  MISMATCH (818018 elems)   <- never ran
53  40x40x256    0.03 ms  MISMATCH (409047 elems)   <- never ran
83  20x20x512    0.03 ms  MISMATCH (204394 elems)   <- never ran
```
and the CU then wedged at `ap_ctrl = 0x203` (ap_start + ap_done, never
idle). `xmutil unloadapp; xmutil loadapp` cleans that up; do NOT try to
JTAG-reset it (§25).

Correct sequence for a v++ kernel:
```
wait ap_idle -> write args -> write ap_start(0x1)
             -> poll ap_done(0x2) -> write ap_continue(0x10)
```
**If the nine IP-flow cores are ever rebuilt through the kernel flow, their
driver must be updated the same way.**

### 32.4 *** MEASURED: LCAM GATE IN THE HYBRID BITSTREAM ***
`kv260_hybrid_vitis/package/bench_lcam_hybrid.py`, CU at 0xa0020000,
buffers from ZoclBuffer (all in low DDR, §26 assertion re-checked):
```
layer  shape           HW (ms)   numpy ref (ms)   speedup   correct
23     160x160x64         1.21           86.24     71.6x    OK
41     80x80x128          0.40           51.49    129.5x    OK
53     40x40x256          0.15           20.33    134.1x    OK
83     20x20x512          0.07            7.68    114.4x    OK
-----------------------------------------------------------------
TOTAL                     1.82          165.75     91.0x    ALL EXACT
```

**Comparison against every prior measurement of the same work:**
```
original custom IP (512-bit, 100 MHz, standalone)  : 40.60 ms   22.3x
optimised IP       (512-bit, 100 MHz, standalone)  :  3.50 ms    1.92x
*** hybrid IP      (128-bit, 300 MHz, WITH DPU)    :  1.82 ms ***
real VART CPU fallback for these gates             : 83.20 ms   45.7x
```
So the gate is now **45.7x faster than the CPU fallback it replaces**, and
still bit-exact. Note the 1.92x over the standalone build is exactly the
§31.4 prediction: same bytes, but 300 MHz instead of 100 MHz on a bus that
now matches the port width instead of being throttled by a width converter.

Layer 23 alone gives the sustained rate: 3.30 MB in 1.21 ms = **2.73 GB/s**.

**Be careful which CPU baseline is quoted (three exist, do not mix):**
82.1 ms (simple numpy, understates), ~138-166 ms (numpy exact rounding --
this script's own column, varies with load), and **83.20 ms (the real VART
CPU fallback -- the honest denominator for any end-to-end claim).**

### 32.5 OPEN: DPU fingerprint does not match either xmodel
End-to-end inference is blocked on exactly one thing. VART checks the
fingerprint and refuses cleanly (it does NOT hang):
```
CHECK fingerprint fail! model_fingerprint 0x101000056010407 is un-matched
with actual dpu_fingerprint 0x101000012010407.
```
Measured with `package/check_fingerprints.py`:
```
HARDWARE (kv260-hybrid2)  0x101000012010407
lcam_v5.xmodel            0x101000056010407   no   <- stock KV260 config
lcam_v5_2p5.xmodel        0x101000016010407   no   <- one nibble away
```
Aligned, the three differ in only two nibble positions:
```
0101000012010407  hardware
0101000016010407  lcam_v5_2p5   (differs in ONE nibble: 2 vs 6)
0101000056010407  lcam_v5       (differs in TWO: 1 vs 5, 2 vs 6)
```
The Vitis `dpu_conf.vh` used here has `CHANNEL_AUGMENTATION_DISABLE` and
`RAM_USAGE_LOW`, which is NOT the stock KV260 configuration -- that is the
most likely source of the difference. Two ways to close it:

- **(A) Recompile the xmodel** for `0x101000012010407`. This is what the
  error message itself recommends, needs no bitstream rebuild (minutes,
  not ~1.5 h), and the arch.json is already generated at
  `kv260_hybrid_vitis/tmp/link/vivado/vpl/prj/prj.gen/sources_1/bd/top/ip/top_DPUCZDX8G_1_0/arch.json`
  -> `{"fingerprint":"0x101000012010407"}`. Requires the QUANTIZED xmodel
  (the vai_c_xir input) -- not on the Windows PC, presumably in the WSL
  Vitis-AI workspace. **Check this first.**
- **(B) Rebuild dpu.xo** with `CHANNEL_AUGMENTATION_ENABLE` (and possibly
  `RAM_USAGE_HIGH`) to match the stock KV260 fingerprint
  `0x101000056010407`, so the ORIGINAL `lcam_v5.xmodel` runs unmodified.
  Costs a full re-link (~1 h) but leaves the DPU half bit-identical to the
  vendor reference, which is the cleaner story for the paper.

Nothing about §32.4 depends on this: the LCAM measurement is complete and
does not involve the DPU.

---

## 33. *** END-TO-END ON THE HYBRID: DPU RUNS, AND THE CPU COST IS NOW ATTRIBUTED BY MEASUREMENT *** (2026-08-30)

Continues §32. The DPU was made to execute, the whole model was run
end-to-end on the hybrid bitstream, and the 385 ms of CPU fallback was
broken down op-by-op with real timings instead of estimates.

### 33.1 The xmodel already existed
The hybrid DPU's fingerprint is `0x101000012010407`. Neither board xmodel
matched -- but a model compiled for exactly that fingerprint was already
sitting in WSL at
`~/wildfire_project/compiled_hybrid/lcam_hybrid.xmodel`
(`meta.json` -> `"target": "DPUCZDX8G_ISA1_B4096_0101000012010407"`,
`arch_hybrid/arch.json` -> `{"fingerprint":"0x101000012010407"}`).
Copied to the board; `check_fingerprints.py` reports MATCH. No recompile
and no DPU rebuild were needed.

Recompile path, if the DPU config ever changes again: the Vitis-AI 3.0
docker image `xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106` is
installed in WSL, and the quantised inputs are at
`~/wildfire_project/exports/detector_v5_2p5/LCAMYOLOXDetector_int.xmodel`
(and `detector_3p0_v5/` for lcam_v5). Command form is in `.bash_history`:
`vai_c_xir --xmodel <int.xmodel> --arch <arch.json> --output_dir <d> --net_name <n>`

### 33.2 *** THE DPU HANGS WERE A MISSING INTERRUPT, NOT A BROKEN DPU ***
First run after the fingerprint matched still failed:
```
cu timeout! device_core_idx 0 ... state 1, is_done 0
dpu timeout! core_idx = 0   LSTART 0 LEND 0 CSTART 0 CEND 0 ... CYCLE_L 3003063189
zocl: kds_del_cu_context: Domain(0) CU(1) hangs, please reset device
```
`probe_dpu_hang.py` settled it by watching the DPU's own ap_ctrl from a
second process while XRT ran an inference in a child:
```
t=0.00s  ap_ctrl = 0x4   idle
t=1.41s  ap_ctrl = 0x6   done=1 idle=1     <-- THE DPU COMPLETED
LSTART 4571 LEND 4571 CSTART 807 CEND 807 SSTART 1809 SEND 1809 CYCLE_L 2071580
GIC 121 delta 0,  GIC 122 delta 0          <-- but NO interrupt ever fired
```
So the DPU computes correctly; only the completion notification is lost.

Cause, confirmed from the linked block design: v++ cascades **every** CU
interrupt through the platform's `axi_intc_0` into ONE PS line --
`DPUCZDX8G_1/interrupt -> ..._interrupt_concat/In1 -> axi_intc_0/intr ->
ps_e/pl_ps_irq0` -- while zocl registers one direct GIC line **per CU**
(`zocl_irq_intc` at GIC 121 and 122 = SPI 89/90). Nothing programs the
axi_intc's enable registers, so no interrupt propagates at all.

**FIX (no rebuild): put XRT in polling mode.**
```
/home/root/xrt.ini:
    [Runtime]
    ert_polling=true
run with  XRT_INI_PATH=/home/root/xrt.ini
```
`Runtime.ert_polling` is a real key in libxrt_core (verified by strings).
With it, KDS polls the CU status register and the missing IRQ is
irrelevant. The DPU then runs: **subgraph 0 = 8.73 ms steady state**, IRQ
counters still 0, proving polling did the work.

**A first probe run gave a WRONG verdict** and is worth recording: the DPU
was still wedged from the previous timeout (`ap_ctrl = 0x1`, start latched,
never idle), so the child aborted at `xclExecBuf` without submitting
anything and the probe concluded "never completed". **Always
`xmutil unloadapp; xmutil loadapp` before a DPU experiment** and confirm
ap_ctrl reads 0x4 first.

### 33.3 *** END-TO-END, MEASURED ON THE HYBRID BITSTREAM ***
`e2e_graphrunner.py --xmodel lcam_hybrid.xmodel`, real image, 10 runs:
```
mean 426.51 ms   median 426.20   min/max 424.96/428.34   std 0.98
FPS 2.34
```
This reproduces the §29 baseline (430.08 ms / 2.33 FPS) almost exactly, so
the hybrid bitstream costs the DPU nothing.

`profile_dpu_vs_cpu.py`, 8 DPU subgraphs:
```
0  [1,160,160,64]   8.717     4  [1,80,80,2]   16.502
1  [1,80,80,128]    5.926     5  [1,20,20,7]    0.215
2  [1,40,40,256]    3.943     6  [1,40,40,7]    0.314
3  [1,20,20,512]    4.746     7  [1,80,80,7]    0.792
TOTAL DPU        :  41.15 ms
END-TO-END       : 426.51 ms
=> CPU FALLBACK  : 385.36 ms   (90.4%)
```

### 33.4 *** WHERE THE 385 ms ACTUALLY GOES (measured, not estimated) ***
Three ways of attributing it disagreed, so it had to be measured:
- element-count proxy (`enumerate_cpu_ops.py`) implied the gate path was
  ~93% of CPU work (mul 3.07M + fix2float 3.25M + float2fix 3.16M elements
  out of 9.85M);
- a numpy re-implementation of the gate chain costs only 149 ms of 385 ms;
- neither is what VART's C++ CPU runner actually spends.

`DEEPHI_PROFILING=1` makes `cpu_task.cpp` log every CPU op with a
microsecond timestamp. `profile_cpu_ops.py` parses that log and
differences consecutive timestamps. Over 9 frames:
```
THE FOUR LCAM GATES                     ms/frame
  lcam2  (160x160x64)                    103.89
  lcam3  ( 80x80x128)                     54.79
  lcam4  ( 40x40x256)                     27.68
  lcam5  ( 20x20x512)                     13.94
  TOTAL                                  200.30

EVERYTHING ELSE ON THE CPU               ms/frame
  27059           (final [1,8400,7])      91.23
  26612_sink_transpose_0 (80x80x7)        72.48
  26809_sink_transpose_1 (40x40x7)        18.24
  output_sink_transpose_2 (20x20x7)        4.69
  ~30 smaller ops                          ~6.5
  TOTAL                                  ~193
```
Cross-check: 200.30 + 193 = 393 ms vs the 385.36 ms measured by
subtraction -- agrees to ~2%, the difference being the small amount of DPU
time that falls inside a gate's timestamp interval.

**So the LCAM gates are 52% of the CPU fallback, NOT the ~94% previously
assumed.** The earlier "93.9% of CPU work" figure was an element-count
proxy and it overstates the gates' share of TIME. Any end-to-end claim
must use the measured 200.30 ms.

### 33.5 Honest end-to-end projection for the LCAM IP alone
```
end-to-end now                     426.51 ms   2.34 FPS
four gates on the CPU              200.30 ms   (measured)
the same four gates on our IP        1.82 ms   (measured, bit-exact)
=> projected end-to-end            228.03 ms   4.39 FPS   1.87x
```
Caveat: this substitutes measured-for-measured but assumes no extra
data-movement cost between the DPU's output buffer and the IP's input
buffer. Both live in low DDR, so a zero-copy hand-off is possible, but it
is not yet built or measured. Treat 1.87x as a projection until the real
DPU->IP->DPU pipeline runs.

### 33.6 The next bottleneck is already identified -- and we already have IPs for it
With the gates moved to the PL, the YOLOX head becomes the bottleneck:
```
27059 (final fix/transpose/fix2float)   91.23 ms
26612_sink_transpose_0                  72.48 ms
26809_sink_transpose_1                  18.24 ms
output_sink_transpose_2                  4.69 ms
                                       ------- ~187 ms
```
These are transpose / reshape / concat / sigmoid ops -- exactly what
`head_transpose`, `head_concat_reshape` and `head_sigmoid` do, and all
three are already validated on silicon (9/9, §26). They are small tensors
(58,800 elements for the biggest) that the CPU is handling absurdly slowly.

If both the gates and the head ops move to the PL:
```
41.15 (DPU) + 1.82 (LCAM IP) + a few ms (head IPs) + ~6 ms (residual CPU)
  ~= 50-55 ms  ->  ~18-20 FPS
```
That is the design target, and it is now backed by per-op measurements
rather than assumption. It also comfortably exceeds the reference paper's
~15 FPS.

### 33.7 Operational notes added this session
- `XRT_INI_PATH=/home/root/xrt.ini` with `ert_polling=true` is REQUIRED for
  any DPU work on `kv260-hybrid2`. Without it every DPU call times out
  after 10 s and leaves the CU wedged.
- After any DPU timeout, `xmutil unloadapp; xmutil loadapp kv260-hybrid2`
  before the next attempt, and check the CU reads ap_ctrl 0x4.
- The old failed Vivado-flow app `kv260-hybrid` was deliberately left in
  place; the Vitis build installs as **`kv260-hybrid2`** so the two
  bitstream/xclbin pairs can never be mixed (§29.7).

---

## 34. *** THE HEAD DOES NOT NEED CUSTOM IPs -- VART's CPU OVERHEAD IS THE COST *** (2026-08-30)

§33.6 concluded that the YOLOX head (~193 ms/frame) was the next bottleneck
and pointed at `head_transpose` / `head_concat_reshape` / `head_sigmoid`.
**That conclusion was wrong about the remedy.** Measuring first avoided a
platform rebuild and a re-link for nothing.

The suspicious numbers: VART spends 91.23 ms transposing a 58,800-element
tensor and 72.48 ms on a 44,800-element one. That is not a plausible amount
of arithmetic. `measure_head_numpy.py` times the identical ops in numpy on
the same ARM cores:
```
op                                        elements   numpy ms   VART ms   ratio
sink_transpose 20x20x7                       2,800      0.381      1.17      3x
sink_transpose 40x40x7                      11,200      0.636      4.56      7x
sink_transpose 80x80x7                      44,800      2.271     18.12      8x
concat+reshape -> [1,7,8400]                58,800      0.059      0.10      2x
final [1,8400,7] fix/transpose/fix2float    58,800      1.339     91.23     68x
sigmoid 20x20x2                                800      0.387      0.05    0.1x
sigmoid 40x40x2                              3,200      0.610      0.05    0.1x
sigmoid 80x80x2                             12,800      1.486      0.05    0.03x
--------------------------------------------------------------------------------
TOTAL                                                   7.168    115.33
```
So the head cost is **VART per-op overhead**, not computation. A
hand-written pipeline replaces those ops with numpy anyway, so the ~193 ms
largely evaporates for free.

**Note the sigmoids go the OTHER way** -- VART is 10-30x FASTER than numpy
on the small [1,H,W,2] sigmoids (0.05 ms vs 0.4-1.5 ms), presumably a
lookup table. Do not blanket-replace CPU ops with numpy; keep VART's
sigmoid or write a LUT.

### 34.1 Revised projection -- NO bitstream rebuild required
```
41.15 (DPU, 8 subgraphs, incl. per-subgraph submit/wait)
 1.82 (LCAM IP, measured, bit-exact)
 7.17 (numpy glue for the head)
-----
50.14 ms  ->  19.9 FPS      (vs 426.51 ms / 2.34 FPS today)
```
The DPU figure already includes `execute_async`+`wait` per subgraph
(measured that way in `profile_dpu_vs_cpu.py`), so scheduling overhead is
counted. NOT counted: buffer hand-off between subgraphs and into the LCAM
CU. If that costs another ~20 ms the result is ~70 ms -> ~14 FPS, i.e.
still at or near the reference paper's ~15 FPS.

### 34.2 What this changes about the plan
- `head_transpose`, `head_concat_reshape`, `head_sigmoid` **do not need to
  go into the bitstream**. The current `kv260-hybrid2` bitstream is
  sufficient to reach ~20 FPS. No platform rebuild, no re-link.
- The remaining work is entirely SOFTWARE: a hand-written runner that
  drives the 8 DPU subgraphs, calls the LCAM CU at the four gates, and
  does the glue in numpy.
- The one real technical risk is the zero-copy hand-off: getting the
  physical address of VART's tensor buffers so the LCAM CU can read/write
  them in place instead of copying 3 MB per gate.

---

## 35. *** END-TO-END HYBRID PIPELINE RUNS: 94.07 ms, 10.63 FPS, 4.53x *** (2026-08-30)

The DPU -> custom-IP -> DPU pipeline of §33.5 is BUILT AND MEASURED. It is no
longer a projection.

`vitis_hybrid/package/hybrid_pipeline.py` replaces GraphRunner: it walks all
24 subgraphs of `lcam_hybrid.xmodel` in topological order, runs the 8 DPU
subgraphs with `vart.Runner`, executes the 15 CPU subgraphs itself in numpy,
and sends the four LCAM attention gates to the CU at 0xa0020000.

```
                        gates=ip     gates=numpy   GraphRunner
DPU (8 subgraphs)         42.81 ms      43.15 ms         41.15
four LCAM gates           26.05 ms     (in CPU col)     200.30
other CPU subgraphs       21.03 ms     250.53 ms        ~193
----------------------------------------------------------------
TOTAL                     94.07 ms     295.75 ms        426.51 ms
FPS                       10.63          3.38             2.34
speedup vs GraphRunner     4.53x         1.44x            1.00x
```

**Two independent wins, and it is worth separating them:**
- moving the four gates to the custom CU: 200.30 -> 26.05 ms
- replacing VART's CPU ops with numpy: ~193 -> 21.03 ms (§34, free)

### 35.1 *** CORRECTNESS: THE IP IS BIT-EXACT END-TO-END ***
Running the SAME pipeline with `--gates numpy` and `--gates ip` and diffing
the final [1,8400,7] output (`ab_compare.py`):
```
max|diff| = 0    mean = 0    mismatched elems = 0 of 58800
```
Every other line of code is identical between the two runs, so this isolates
the IP exactly. **The custom accelerator reproduces the software reference
bit for bit through the whole network**, not just on synthetic tensors.

### 35.2 Why the pipeline differs slightly from GraphRunner (and why that is fine)
Against `e2e_outputs.npz` the pipeline shows `max|diff| = 0.21875`
(7 LSB at fix_point=5), mean 0.0142 (~0.45 LSB). Two causes, NEITHER of
them the IP:
1. **Input rounding.** fix_point is -1, so quantising is pixel/2 and every
   ODD pixel lands exactly on .5 -- where `np.round` (ties-to-even, what
   e2e_graphrunner.py used) and DPU_ROUND (ties-away-from-zero, correct)
   disagree. `--round ref` reproduces the old behaviour; using DPU_ROUND
   doubled the diff to 0.4375, confirming this is the dominant term.
2. **VART's sigmoid is a lookup table.** It does 12,800 elements in
   0.05 ms -- far too fast for expf, and faster than numpy (§34). With an
   int8 input there are only 256 possible values, so a LUT is exact for
   the quantised domain but differs from evaluating sigmoid in float.

Both are preprocessing/approximation differences of well under 1 LSB on
average. §35.1 is the correctness claim that matters for the IP.

### 35.3 *** THE REMAINING BOTTLENECK IS AN UNCACHED READ ***
Phase timing inside the gate call:
```
write into buffer + sync_to_device :  2.15 ms   (3.07 MB -> 1.4 GB/s)
CU execution                       :  2.18 ms   (cf. 1.82 ms standalone)
sync_from_device ioctl             :  0.10 ms   (nearly free)
read back out of the buffer        : 21.52 ms   (3.07 MB -> 143 MB/s)
```
**Reads out of the zocl mapping are 10x slower than writes into it.** That
asymmetry is the signature of a WRITE-COMBINING mapping: writes coalesce in
the write buffer, reads go uncached to DRAM with no prefetch. The cache
ioctl is NOT the cost -- 0.10 ms -- so this is the mapping's attributes.

Two earlier guesses were WRONG and are recorded so they are not retried:
- "the 8 MB sync is the cost" -- shrinking buffers 8 MB -> 2 MB and syncing
  only the used bytes moved 29.79 -> 26.32 ms. Real but minor.
- "write_array's double copy is the cost" -- writing through a
  `np.frombuffer` view helped only marginally. Writes were never the problem.

### 35.4 NEXT ACTION (first thing next session)
`hw_ip_driver.py` allocates with `ZOCL_BO_FLAGS_CMA = 0x1 << 28` only. zocl
also defines a CACHEABLE flag; allocating the OUTPUT buffer cacheable should
make reads run at write speed, and the `sync_from_device` call already in
place would then do the necessary invalidate.

**Do NOT guess the flag's bit position** -- read it from the real
`zynq_ioctl.h` for XRT 2.14 / 2022.2. A wrong flag risks a bad allocation.

If the read drops to write speed (~2.15 ms):
```
gates = 2.15 + 2.18 + 0.10 + 2.15 = 6.58 ms
total = 94.07 - 26.05 + 6.58     = 74.6 ms  ->  13.4 FPS
```
Further headroom after that: `other CPU subgraphs` is 21.03 ms against a
7.17 ms numpy floor (§34) -- the gap is the exact-sigmoid evaluation, which
should become a 256-entry LUT like VART's.

Realistic landing zone: **~65-75 ms, 13-15 FPS**, at the reference paper's
~15 FPS, from a bitstream that already exists.

---

## 36. *** 75.52 ms / 13.24 FPS / 5.65x -- AND TWO PROVABLY-EXACT CPU OPTIMISATIONS *** (2026-08-30)

Continues §35 in the same session. Three things were done: the "cacheable
buffer" plan from §35.4 was tested and KILLED, and two exact optimisations
took the CPU side from 20.92 ms to 2.70 ms per frame.

```
                     §35        §36        GraphRunner
DPU (8 subgraphs)   42.81      43.23           41.15
four LCAM gates     26.05      26.03          200.30
other CPU           20.96       2.70          ~193
--------------------------------------------------------
TOTAL               94.07      75.52 ms       426.51 ms
FPS                 10.63      13.24            2.34
speedup              4.53x      5.65x           1.00x
```

### 36.1 THE §35.4 PLAN WAS WRONG -- there is no CACHEABLE flag
§35.4 said to allocate the output buffer "cacheable" using a zocl flag. The
real `zynq_ioctl.h` for XRT 2022.2 (Xilinx/XRT, src/runtime_src/core/edge/
include/zynq_ioctl.h -- note: NOT under drm/zocl/, that path 404s) defines
only:
```
DRM_ZOCL_BO_FLAGS_HOST_BO   (0x1 << 26)
DRM_ZOCL_BO_FLAGS_COHERENT  (0x1 << 27)
DRM_ZOCL_BO_FLAGS_CMA       (0x1 << 28)
DRM_ZOCL_BO_FLAGS_SVM       (0x1 << 29)
DRM_ZOCL_BO_FLAGS_USERPTR   (0x1 << 30)
DRM_ZOCL_BO_FLAGS_EXECBUF   (0x1 << 31)
```
**There is no CACHEABLE flag.** COHERENT was the only candidate, and
`bench_bo_flags.py` measured it directly:
```
CMA (current)   phys=0x017100000 LOW-DDR  write 0.63 ms (2591 MB/s)  read 12.12 ms (135 MB/s)
CMA|COHERENT    phys=0x017100000 LOW-DDR  write 0.71 ms (2311 MB/s)  read 12.09 ms (135 MB/s)
COHERENT only   phys=0x017100000 LOW-DDR  write 0.68 ms (2409 MB/s)  read 12.09 ms (136 MB/s)
```
Identical. The mapping is write-combining regardless of flag. **Do not
retry allocation flags for this.** The 5-minute isolated benchmark was worth
far more than changing the pipeline on the assumption.

`probe_vart_api.py` also settled the zero-copy route: `vart.RunnerExt` exists
with `get_inputs()`/`get_outputs()`, but `vart.TensorBuffer` exposes ONLY
`get_tensor()` -- no `data()` or `data_phy()`. So Python cannot obtain a
device address. True zero-copy needs either a `/proc/self/pagemap` lookup on
the TensorBuffer's virtual address, or the C++ API.

### 36.2 OPTIMISATION 1 -- collapse elementwise subgraphs to a 256-entry LUT
Six CPU subgraphs are `fix2float -> sigmoid -> float2fix` over ONE int8
tensor. int8 has 256 possible values and every op is elementwise, so the
whole chain IS a 256-entry int8->int8 table, built once at startup by
running the same numpy ops over all 256 inputs (exact by construction).

This is exactly what VART does, and why VART beat numpy on these ops (§34):
subgraph [10] was spending **1.17 ms on a 400-element tensor** -- essentially
all per-call overhead.
```
six sigmoid subgraphs:  ~7.3 ms  ->  ~0.6 ms
```

### 36.3 OPTIMISATION 2 -- strip the pointless float round-trip
Four subgraphs are `fix2float(fp) -> transpose -> float2fix(fp)` (or `-> fix(fp)`)
with **the same fix_point on both ends**. Dequantising by 2^-fp and
requantising by 2^fp is exactly the identity for int8: both are exact powers
of two, float32 represents every int8 exactly, and nothing can leave
[-128,127]. So the subgraph is just a transpose -- of int8, not float32,
i.e. a quarter of the bytes and none of the temporaries.
```
[23] 5.58 -> 0.95    [21] 3.97 -> 0.15
[18] 1.54 -> 0.09    [15] 1.19 -> 0.09
```
Guarded by an explicit `fp_in == fp_out` check; where the scales genuinely
differ the round-trip is real and the slow path is kept.

Combined: **other CPU 20.92 -> 2.70 ms**, below the 7.17 ms numpy floor
estimated in §34, because that estimate still assumed the float round-trip.

### 36.4 CORRECTNESS -- both optimisations are exact, and so is the IP
```
optimised vs --nolut control : max|diff| = 0, 0 of 58800 elems differ
gates=ip   vs gates=numpy    : max|diff| = 0, 0 of 58800 elems differ
```
The first proves the two CPU optimisations changed nothing numerically. The
second (from §35.1) proves the custom accelerator matches software exactly.

### 36.5 WHAT IS LEFT: the uncached read-back, now 29% of the frame
```
DPU                     43.23 ms   57%
gates                   26.03 ms   34%   <- of which read-back 21.57 ms
other CPU                2.70 ms    4%
```
The single biggest remaining item is copying the CU's output out of the
write-combining mapping: 3.07 MB at ~140 MB/s. Everything else in the gate
is already fast (write 2.11, CU run 2.18, sync 0.10).

Solving it lands at **~54 ms -> ~18.5 FPS**. The two viable routes, now that
allocation flags are ruled out:
1. `/proc/self/pagemap` on a `RunnerExt` TensorBuffer's virtual address to
   recover its physical address, then point the CU's `feat_out` straight at
   the next DPU subgraph's input buffer. CMA pages are pinned, so the
   translation is stable.
2. Drop to the C++ VART API, where `TensorBuffer::data_phy()` exists.

**Current standing: 13.24 FPS, measured, bit-exact, on one bitstream.**
Close to the reference paper's ~15 FPS, with a clear route past it.

---

## 37. *** THE READ-BACK IS A PLATFORM LIMIT: TWO ROUTES TRIED, BOTH DEAD *** (2026-08-30)

§36 left one item: the 21.57 ms spent copying the LCAM CU's output out of a
write-combining mapping, 29% of a 75.5 ms frame. Both plausible fixes were
tried and both failed. **Standing result is unchanged and correct:
75.75 ms, 13.20 FPS, 5.63x, bit-exact.**

### 37.1 ROUTE 1 -- zero-copy into VART's buffer: DEAD
Plan: get the physical address of the next DPU subgraph's input TensorBuffer
via /proc/self/pagemap, and point the CU's feat_out at it.

`spike_zerocopy.py` measured it:
```
RunnerExt.get_inputs() -> writable numpy view      OK
virtual  addr = 0xaaaab41f6ad0                     (a malloc address)
physical addr = 0x817968ad0   -> HIGH DDR, PL cannot reach it
first 8 pages physically contiguous: False
```
**VART's TensorBuffers are ordinary pageable host heap, not CMA.** Virtually
contiguous, physically scattered, and above 0x8_0000_0000. VART copies
host->device internally at execute time. There is nothing to point the CU at.
The C++ API would not help: it is the same buffer.

### 37.2 *** A SAFETY BUG IN MY OWN TEST -- read this before writing another ***
The first version of `spike_zerocopy.py` PRINTED "HIGH DDR -- PL CANNOT
REACH" and "contiguous: False" and then **triggered the CU anyway**. The
accelerator wrote 1.6 MB into scattered physical pages belonging to other
allocations; the process died with `double free or corruption (out)`.

Only 1216 of 1638400 bytes landed where expected -- the rest went into
whatever else lived in those pages.

**Never make an address check advisory when the consequence is a DMA write.**
The script now returns early with "REFUSING to run the CU" on either
condition. Any future experiment that hands a raw physical address to a
bus master must do the same.

### 37.3 ROUTE 2 -- cached second mapping: FAST BUT INCOHERENT, REVERTED
The buffer is ordinary DDR; the slowness is the MAPPING. Mapping the same
physical pages again through /dev/mem WITHOUT O_SYNC gives a cached mapping.
`bench_cached_map.py`:
```
zocl DRM mmap (current)        12.00 ms    137 MB/s   data OK
/dev/mem  no O_SYNC (cached)    0.67 ms   2432 MB/s   data OK   <- 18x
/dev/mem  with O_SYNC          11.84 ms    138 MB/s   data OK
mmap slice -> bytes            11.80 ms    139 MB/s
np.copyto into preallocated    11.98 ms    137 MB/s
```
Wired into the pipeline it gave **54.83 ms -> 18.24 FPS, 7.78x**.

**And it was WRONG: 10414 of 58800 output elements differed from the numpy
control.** The mapping is not coherent. zocl's `sync_from_device()` is
evidently a no-op for these CMA buffers -- memory allocated coherent needs no
cache maintenance, so the kernel skips it -- leaving stale lines in our
cached view. Reverted to the write-combining read.

### 37.4 *** MY VERIFICATION GAVE A FALSE PASS -- why, and the lesson ***
`verify_cached_read.py` ran 12 iterations across all four gate sizes with
fresh random data each time and reported **12/12 correct, "SAFE to use"**.
It was wrong, and the pipeline caught what it missed:

1. it poisoned the output buffer through the write-combining view and called
   `sync_to_device` before every run -- an extra cache operation the pipeline
   does not perform;
2. it cycled sizes up to 1.638 MB, larger than the 1 MB L2, so ordinary cache
   pressure evicted the stale lines for it.

It never reproduced the pipeline's real access pattern. **A coherency test
that does not replicate the exact sequence of the production path proves
nothing.** The honest check was the end-to-end A/B against the numpy control,
which is cheap and total -- prefer it over a bespoke micro-test next time.

### 37.5 Where this leaves the read
```
DPU                     43.0 ms   57%
gates                   26.0 ms   34%   <- read-back 21.8 ms of it
other CPU                2.7 ms    4%
```
Ideas now exhausted or ruled out: allocation flags (§36.1), zero-copy into
VART buffers (§37.1), cached second mapping (§37.3). What is left is
unattractive: a kernel module exposing a cacheable-but-PL-reachable
allocation, or a udmabuf-style allocator if the image has one. Neither is
worth the risk for ~20 ms.

**Treat 13.20 FPS as the result.** It is 5.63x over the GraphRunner baseline,
bit-exact, on one bitstream, and the remaining cost is a documented platform
property rather than an unexplained gap.

---

## 38. *** THE THREE OPEN ITEMS CLOSED: ACCURACY, POWER, PIPELINING *** (2026-08-30)

§37 listed three things as "not established". All three are now measured.

### 38.1 *** ACCURACY: mAP 0.7360 @0.5, AND THE ACCELERATOR COSTS NOTHING ***
A labelled validation split exists after all -- `dataset/extracted/data/val`
in WSL holds 3099 images WITH 3099 YOLO-format label files (classes 0=fire,
1=smoke). A deterministic 400-image slice (every 7th of the sorted list, 208
fire + 292 smoke objects) was copied to the board and scored with
`eval_map.py`: AP at IoU 0.5, all-point interpolation, plus mAP@[.5:.05:.95].

```
                        gates = IP            gates = software (control)
class     objects   AP@0.5   AP@.5:.95     AP@0.5   AP@.5:.95
fire          208   0.7378      0.4103      0.7378      0.4103
smoke         292   0.7342      0.3446      0.7342      0.3446
------------------------------------------------------------
mAP                 0.7360      0.3774      0.7360      0.3774
detections            664                     664
latency          75.84 ms                289.87 ms
```
**Identical to four decimal places, same 664 detections.** The bit-exactness
of §35.1 now has a dataset behind it, not one demo image: moving the four
attention gates to custom silicon changes no detection, no score, no box.
Within the same pipeline the accelerator is worth **3.82x** on latency
(75.84 vs 289.87 ms).

Latency was flat over all 400 frames (75.8 ms at every 50-frame checkpoint),
so the single-image figure was not a lucky sample.

Caveat kept: this is a 400-image slice, not the full 3099, and it is a
val split -- no separate held-out test set was scored.

### 38.2 *** POWER: MEASURED 6.36 W -- AND VIVADO'S ESTIMATE IS PROVABLY HIGH ***
The KV260 SOM carries an INA260 on its input rail, exposed at
`/sys/class/hwmon/hwmon0` (`name` = `ina260_u14`) with `power1_input` in
microwatts. That is a real measurement, unlike `report_power`.

```
MEASURED (INA260, whole module: PS + PL + DDR)
  idle, design loaded      4.809 W    (0.960 A @ 5.007 V)
  running the pipeline     6.360 W    mean, 10.110 W peak
  delta = the workload     1.551 W

VIVADO report_power on the routed checkpoint (vectorless, confidence MEDIUM)
  Total On-Chip            6.896 W    dynamic 6.561 + static 0.335
  PS8 (modelled)           2.456 W    Signals 1.214, DSPs 0.701,
                                      CLB 0.644, Clocks 0.867,
                                      URAM 0.329, BRAM 0.251
  Junction temperature     41.0 C
```
**Vivado claims the chip alone burns 6.896 W while the entire module
measurably draws 6.360 W at its input.** That is physically impossible -- the
input rail must exceed on-chip consumption plus regulator loss -- so the
vectorless estimate is demonstrably too high. It assumes default toggle rates
the design does not actually hit. **Quote the measured 6.36 W.**

#### Energy per frame -- the honest efficiency metric
Instantaneous power alone would flatter the slow baseline. Both paths were
run under the meter (`measure_power_baseline.py`), same bitstream, differing
only in software:
```
                       power    latency   energy/frame
stock GraphRunner     5.339 W  427.70 ms     2.283 J
hybrid pipeline       5.794 W   75.97 ms     0.440 J
                      ---------------------------------
                       0.92x      5.63x        5.19x
```
The hybrid draws 8.5% MORE instantaneous power and is **5.19x more energy
efficient per inference**, because it finishes far sooner. For a battery
powered UAV that ratio is the result, not the watts.

### 38.3 *** PIPELINING: 19.43 FPS, ABOVE THE REFERENCE PAPER'S ~15 ***
Feasibility was checked before building anything (`probe_gil.py`), because
Python threads only help if the GIL is released:
```
operation                        1 thread  2 threads   gain
numpy copy from WC mapping        11.8 ms     6.8 ms   1.72x
DPU subgraph (VART)                8.9 ms     7.3 ms   1.22x
numpy matmul (control)           708.4 ms   355.1 ms   2.00x
```
The DPU's 1.22x is CORRECT, not a failure: it is a single hardware unit, so
two threads cannot execute it concurrently -- that gain is only submit
overhead overlapping. What mattered is that the non-DPU work releases the GIL
(1.72x) and can therefore overlap another frame's DPU execution.

`pipelined_throughput.py` runs N worker threads, each with its OWN vart.Runner
set (runners are not thread-safe) and a SHARED LcamCU behind a lock -- one
physical accelerator with one set of buffers, so two threads driving it would
overwrite each other's input mid-flight.
```
workers   throughput      FPS    per-frame latency
   1       76.29 ms     13.11         76.18 ms
   2       53.39 ms     18.73        106.57 ms     <- operating point
   3       51.93 ms     19.26        151.91 ms
   4       51.48 ms     19.43        205.55 ms
```
Saturates near 19.4 FPS. **Two workers is the sensible setting**: beyond it
you buy 0.7 FPS for 100 ms of extra latency.

The floor is ~51.5 ms rather than the DPU's 43 ms because the LCAM CU is a
second serial resource -- its lock is held for ~26 ms per frame, most of it
the write-combining read-back (§37.5). Fixing that read would lift both the
single-frame and the pipelined numbers.

**Latency does NOT improve -- it gets worse.** Pipelining is a throughput
result: 13.19 -> 18.73 FPS while per-frame latency goes 75.8 -> 106.6 ms.
For a video feed that is the right trade; for single-shot latency it is not.
State which one is being claimed.

### 38.4 Standing results
```
end-to-end latency, single frame   75.84 ms     13.19 FPS    5.62x
throughput, 2 workers              53.39 ms     18.73 FPS    8.00x
peak throughput, 4 workers         51.48 ms     19.43 FPS    8.30x
mAP@0.5 / mAP@.5:.95                0.7360 / 0.3774  (unchanged by the IP)
board power under load              6.360 W measured
energy per frame                    0.440 J      5.19x better
```

---

## 39. *** BRAM 98.5 -> 81 (-17.8%), TIMING STILL MET, FINGERPRINT UNCHANGED *** (2026-08-30)

A resource-optimised rebuild, in NEW folders throughout -- `kv260_hybrid_vitis`,
its bitstream and the board's `kv260-hybrid2` app are all untouched and still
the verified fallback. New build lives in `C:\Xilinx\projects\kv260_hybrid_opt`,
installs as **`kv260-hybrid3`**.

### 39.1 Exact utilisation, before and after
```
                       BEFORE (hybrid2)        AFTER (hybrid3)
CLB LUTs            61,903 / 117,120 52.85%   61,554  52.56%
CLB Registers      110,807 / 234,240 47.30%  111,010  47.39%
CLB slices          14,201 /  14,640 97.00%   14,018  95.75%   <- more headroom
Block RAM Tile        98.5 /     144 68.40%       81  56.25%   <- -17.5 tiles
URAM                    46 /      64 71.88%       50  78.13%
DSP48E2                716 /   1,248 57.37%      716  57.37%

per kernel                BRAM  URAM   LUT        BRAM  URAM   LUT
  DPUCZDX8G_1               82    46  48,996        67    50  48,932
  lcam_attention_gate_opt_1   7     0  10,062         4     0   9,771
  platform                    9     0   2,852         9     0   2,861

WNS  +0.031 ns -> +0.018 ns    both MET, 0 failing endpoints
```

### 39.2 Where the BRAM went, and why the DPU was the only real lever
83% of the block RAM was the DPU (82 of 98.5), not the custom IP. Two changes:

**DPU: URAM bank counts raised** in `dpu_conf.vh`,
`def_UBANK_IMG_N` 5 -> 7 and `def_UBANK_WGT_N` 17 -> 21 (23 -> 29 banks).
`RAM_USAGE_LOW` was already set, so this was the remaining knob.
Result: **82 -> 67 BRAM for 4 more URAM.** A far better trade than the
~2 URAM/bank ratio predicted -- 15 tiles freed for 4 URAM.

**LCAM IP: AXI adapter FIFOs shrunk** from `num_outstanding=16,
max_burst=64` to `4 / 32`. The kernel's block RAM was never datapath storage
-- csynth reports BRAM_18K = 0 -- it was the m_axi buffers, whose depth is
outstanding x burst (16 x 64 x 128 bit = 16 KB per port). Measured by OOC
synthesis of the generated RTL (`ooc_synth.tcl`), which is the only way to
see them since csynth does not model them:
```
16 x 64 (baseline)   7.0 BRAM   17,991 LUT
 4 x 16 (minimal)    4.5 BRAM   14,793 LUT
```
Chose 4 x 32 rather than 4 x 16: still 128 beats in flight, enough to cover
DDR latency, and it kept Fmax at 411 MHz. Placed result 4 BRAM / 9,771 LUT.

### 39.3 *** THE FINGERPRINT DID NOT CHANGE ***
```
before  0x101000012010407
after   0x101000012010407
```
`lcam_hybrid.xmodel` still loads -- **no recompile, drop-in replacement.**

This resolves an apparent contradiction. `dpu_conf.vh` states that URAM
changes "Don't need update model", while §29.15 recorded the fingerprint
moving 0x...12 -> 0x...16 when URAM was touched. Both are right: enabling
URAM AT ALL changes the arch, adjusting how many BANKS use it does not.

### 39.4 Note on the LUT figure
Device LUT barely moved (61,903 -> 61,554) even though the LCAM kernel shed
~300 placed LUT and ~3,000 by HLS estimate: the DPU grew slightly with the
extra URAM banks and absorbed most of it. The useful win is in SLICES,
97.00% -> 95.75%, which is the binding constraint on this design -- BRAM at
68% never was.

### 39.5 Status: BUILT, NOT YET HARDWARE-VERIFIED
The board was powered off during this work. Package is staged at
`kv260_hybrid_opt\package\out\` (hybrid3.bit.bin, hybrid3.xclbin,
pl_overlay_hybrid3.dts, shell.json, install_hybrid3.sh) and installs
alongside, not over, kv260-hybrid2.

To verify when the board is back:
```
xmutil unloadapp; xmutil loadapp kv260-hybrid3
python3 hybrid_pipeline.py --gates ip --runs 5          # expect ~75.8 ms
python3 eval_map.py --gates ip                          # expect mAP 0.7360
python3 pipelined_throughput.py --workers 2 --frames 40  # expect ~18.7 FPS
```
**What to watch:** the smaller AXI FIFOs are the one change that could cost
throughput. If the gate's CU time rises above ~2.2 ms, revert the FIFO
setting to 8/32 and relink; everything else here is risk-free.

### 39.6 *** HARDWARE-VERIFIED (2026-08-30) -- and one real regression ***
`kv260-hybrid3` loaded and tested. Fingerprint on hardware confirmed
`0x101000012010407`, so `lcam_hybrid.xmodel` loaded unchanged as predicted.

```
                              hybrid2        hybrid3      delta
BRAM tiles                 98.5  68.40%    81   56.25%   -17.5  (-17.8%)
CLB slices                       97.00%          95.75%
URAM                       46    71.88%    50   78.13%
CLB LUTs                   61,903 52.85%   61,554 52.56%
WNS                        +0.031 ns       +0.018 ns     both MET

LCAM CU execution            2.18 ms        2.18 ms      unchanged
single-frame latency        75.88 ms       75.6 ms       unchanged
mAP@0.5 / mAP@.5:.95     0.7360 / 0.3774  0.7360 / 0.3774  identical
IP vs software control    0 / 58,800      0 / 58,800     bit-exact
pipelined, 2 workers        18.73 FPS      18.02 FPS     -3.8%
pipelined, 4 workers        19.43 FPS      19.28 FPS     -0.8%
```

**The AXI FIFO reduction cost nothing in isolation but does cost under
contention.** CU execution is identical at 2.18 ms and single-frame latency
is unchanged (75.66 / 80.58 / 75.57 ms over three repeats), but pipelined
2-worker throughput is reproducibly lower -- 18.02, 18.02, 18.01 FPS across
three runs, so this is signal, not noise.

Mechanism: with `num_outstanding=16` the kernel could absorb DDR contention
while the DPU hammered memory concurrently; at 4 it stalls sooner. That only
manifests when frames overlap, which is exactly why single-frame timing
looks unaffected. Resource savings measured in isolation can still cost
throughput under concurrency -- worth remembering.

**The trade as measured: -17.5 BRAM tiles and -1.25% slices for -0.71 FPS
pipelined.** Accuracy and bit-exactness are untouched either way.

If the throughput matters more than the BRAM, relink with the FIFOs at
`8 / 32` (measured 15,480 LUT at HLS, between the two points tested) --
that should recover most of the contention tolerance while keeping most of
the block-RAM saving. Both bitstreams are installed and independently
loadable, so the choice can be made from data.

**Operational note:** the FIRST DPU run after loading hybrid3 timed out and
had to be recovered with `xmutil unloadapp; xmutil loadapp`. Same transient
as §33.2 -- always reload before the first DPU experiment on a freshly
installed app, and do not read the first timeout as a build failure.

### 39.7 8x32 FIFO DOES NOT FIT -- timing failed (2026-08-30)
§39.6 suggested relinking with the LCAM AXI FIFOs at `8 / 32` to recover the
0.71 FPS lost under pipelining while keeping the block-RAM saving. **It does
not close timing.**
```
DPU URAM   LCAM FIFO     WNS        BRAM   pipelined 2w
   46       16 x 64    +0.031 ns    98.5     18.73 FPS
   50        4 x 32    +0.018 ns      81     18.02 FPS   <- kv260-hybrid3
   50        8 x 32    -0.024 ns FAIL  --        --
```
`-0.024 ns, TNS -3.170, 268 failing endpoints of 361,267`. The extra ~1,500
LUT of the larger FIFOs, on top of the URAM change that already tightened
placement, pushes it past closure at 95.75% slice occupancy.

**The three variables trade against each other and cannot all be maximised:
block RAM, AXI FIFO depth (which buys throughput under contention), and
timing slack.** hybrid3 is the point that satisfies the block-RAM goal while
still closing.

A retry with `Performance_ExploreWithRemap` was launched, on the reasoning
that a -0.024 ns miss at 95.75% occupancy is a placement/congestion problem
rather than logic depth. Low expected value -- it is worth at most 0.71 FPS.
**If it fails, keep kv260-hybrid3 and stop optimising this axis.**

### 39.8 *** BOARD WEDGED IN D STATE -- REQUIRES A HARD POWER CYCLE ***
While hybrid3 was loaded, `measure_power.py` was run against it. Its pipeline
subprocess failed silently and the meter reported a plausible-looking but
meaningless 5.003 W (0.071 W delta, versus 1.551 W on hybrid2) with no
latency line -- the signature of a workload that never ran.

That left the CU wedged, and recovery failed at every level:
```
pid 2055  python3 hybrid_pipeline.py   state D   unkillable
pid 2633  xmutil unloadapp             state S   blocked behind it
pid 3087  xmutil listapps              state S   blocked
load average 2.00
```
The client is in **uninterruptible sleep (D)**, so no signal reaches it --
`kill -9` included -- and it holds the zocl context, so every subsequent
`xmutil` blocks forever. SSH still responds; the PL reload path does not.

**There is no software recovery from D state. Only a hard power cycle.**
`reboot` is not enough: shutdown blocks on the same process. After power-up
the IP must be re-applied (`ip addr add ... / ip route add ...`) -- if those
commands answer "File exists", the board did NOT actually reboot.

Two lessons:
1. `measure_power.py` reported a number without checking that its workload
   ran. **Any harness that wraps a subprocess must fail loudly when the
   subprocess fails**, or it will manufacture a plausible wrong result --
   which is exactly what happened here.
2. §33.2's advice to reload before a DPU experiment is necessary but not
   sufficient: once a client reaches D state, reloading is no longer
   available as a remedy.

### 39.9 WHY THE LCAM KERNEL IS ~10k LUT -- and two optimisations that BACKFIRED
Question raised: with only ONE gate instance, why does it use so much logic?
Per-instance breakdown of the deployed 128-bit / 4x32 kernel:
```
BEAT_LOOP (compute)   8,659 LUT   58%    <- the arithmetic, not the ports
gmem0 m_axi master    1,316 LUT
gmem2 m_axi master    1,316 LUT
gmem1 m_axi master    1,229 LUT
control_s_axi           808 LUT
mux / expression      1,559 LUT
                     --------
                     14,985 LUT (HLS)  ->  9,771 placed
```
AXI infrastructure is only 31%. The compute loop is 58%: 16 lanes at ~540 LUT
each. **The expensive element is that `shift` is a RUNTIME argument, so every
lane needs a full barrel shifter** -- sixteen of them. The 8x8 multiply is
comparatively cheap.

#### Both "obvious" fixes made it WORSE (measured, ablated)
```
variant                       LUT    DSP    vs baseline
uniform 32-bit (deployed)  14,985      6       --
narrow to ap_int<20>       21,584      6     +44%
BIND_OP multiply to DSP    15,883     22      +6%
both                       20,944     22     +40%
```
**HLS already performs bit-width inference.** It sees that `f` and `w_q` come
from `ap_int<8>` casts and narrows the datapath itself. Declaring an explicit
`ap_int<20>` accumulator and casting to `ap_int<32>` for the result forced
width conversions on every lane and defeated that analysis. Hand-narrowing
fought the tool. Binding the multiply to DSP also cost LUT rather than saving
it -- the surrounding muxing outweighed the multiplier moved out.

Equivalence of the narrow version was verified exhaustively first (all 65,536
int8 x int8 pairs, all four layer configs, IDENTICAL, widest intermediate
16,448 vs ap_int<20>'s 524,287) -- so this is a genuine area result, not a
correctness failure.

#### The only lever that would work, and why it is not worth taking
Halving LANES 16 -> 8 would roughly halve the compute loop (~9,771 -> ~7,000
placed LUT) but drop the port from 4.8 to 2.4 GB/s, below the 2.73 GB/s the
kernel currently sustains -- gate time ~2.18 -> ~4 ms. That is 2,800 LUT
(2.4% of device) for +1.8 ms on a 76 ms frame, and it would deepen the
pipelined-throughput regression rather than help it.

**Conclusion: leave the kernel at 9,771 LUT.** It is 8.3% of the device
against the DPU's 42.8%; the custom IP was never where the resources were.
The block-RAM goal was met through the DPU's URAM banks (§39.1), which is
where they actually sat.

---

## 40. *** ALL NINE CUSTOM IPs CONVERTED TO VITIS KERNELS; 10-CU LINK ATTEMPTED *** (2026-08-31)

Question posed: rather than accept that eight of the nine IPs are unused, try
putting the WHOLE custom operator library on the device alongside the DPU and
see what happens. New folders throughout (`hls_work/all9_xo`,
`projects/kv260_all9`); nothing existing touched.

### 40.1 All eight remaining IPs now build as Vitis kernels (NEW)
Until now only `lcam_attention_gate_opt` had ever been through the kernel
flow. All eight others are now converted and exporting `.xo`:
```
conv2d_engine       13,490 LUT   47 DSP        head_sigmoid     7,098 LUT  21 DSP
depthwise_engine    11,805 LUT   36 DSP        head_transpose   5,160 LUT  29 DSP
eltwise_add          6,896 LUT    6 DSP        pool_engine      7,025 LUT   3 DSP
head_concat_reshape  9,557 LUT    0 DSP        upsample_engine 12,257 LUT  31 DSP
                                        TOTAL 73,288 LUT, 173 DSP, 0 BRAM  (at 100 MHz)
```
Two source problems had to be fixed, and both are worth recording:

**(a) The section-23 s_axilite split, in seven of the eight.** Only
`head_transpose` already had the m_axi offsets on the `control` bundle. The
rest hit `[HLS 214-219]`. Fixed by inserting one
`#pragma HLS INTERFACE s_axilite port=<ptr> bundle=control` per m_axi port.
NOTE: `head_concat_reshape` and `head_sigmoid` write their m_axi pragmas
across two lines with a trailing backslash, so a naive "insert after the line
matching m_axi" splits the pragma and produces
`use of undeclared identifier 'offset'`. Insert after the CONTINUATION ends.

**(b) A real latent bug in `depthwise_engine`.** The kernel-flow compiler
rejects
```c
acc_t b = (bias_shift >= 0) ? (acc_t)bias[c] * (acc_t)(1 << bias_shift)
                            : (acc_t)bias[c] >> (-bias_shift);
```
with `[HLS 207-2359] conditional expression is ambiguous`: `ap_int<32> *
ap_int<32>` widens to `ap_int<64>` while the shift branch stays `ap_int<32>`,
so the two ternary arms have different types. The IP flow tolerated it. Fixed
by casting both arms to `acc_t` -- the result was assigned to `acc_t` anyway,
so semantics are unchanged.

### 40.2 *** THREE HARD PLATFORM CONSTRAINTS ON MEMORY PORTS (each cost a link) ***
Getting ten CUs wired taught three rules that are not obvious and are not in
the docs we had:

1. **Do not leave `sp=` unassigned.** v++ defaults kernels onto HPC0 -- the
   DPU's own instruction-fetch port -- and the build dies with
   `[VPL 41-99] Failed to find BusTerm object .../S01_AXI_arcache`.
2. **One memory port accepts at most 16 masters.** All 23 IP masters on HPC1
   gave `[CFGEN 83-2231] Resources exhausted for sp {HPC1}. Resource HPC1 has
   1 interface that can support 16 possible masters.`
3. **The COHERENT ports (HPC0/HPC1) cannot take multiple masters at all.**
   Moving them to HPC1 within the 16 limit still failed, `[VPL 41-99]` on
   `S00_AXI_awcache`: v++ cannot synthesise the per-slave cache/prot tie-offs
   an HPC port requires. `LPD` is declared `S_AXI_HP`-type in our platform and
   has no such restriction.

Working assignment: DPU on HPC0/HP0/HP1 and LCAM on HP2/HP3 exactly as in the
verified design; the eight demonstration IPs split 15 on **LPD** and 8 on
**HP2**. They are idle while the network runs, so sharing HP2 with the LCAM
read port costs nothing in practice.

### 40.3 Status: 10-CU link running
Ten kernels accepted, block design built, **89 block-level synthesis jobs**
(versus 16 for the two-CU design -- the extra is interconnect for 23 more
masters). Projected ~102,600 of 117,120 LUT (88%), requiring slice packing to
improve from today's 4.39 LUT/slice toward ~7.

Clocking is the enabling decision: DPU at 300/600 MHz and the LCAM gate at
300 as before, but **all eight demonstration IPs on the platform's 100 MHz
domain**. The two-CU design closes at only +0.018 ns; asking eight more
kernels for 300 MHz would be hopeless.

Outcome not yet known -- synthesis alone is ~1.5 h at 13/89 jobs per 15 min.
Whether it closes or not is a publishable resource-scaling result: either a
full custom operator library coexists with the vendor DPU on a KV260, or the
limit is quantified.

### 40.4 *** RESULT: THE KV260 HOSTS THE DPU PLUS ~ONE CUSTOM CU, NOT A LIBRARY ***
Three builds were run to find the ceiling. All PC-side, new folders, nothing
existing disturbed.
```
design                LUT           slices          packing      WNS        outcome
DPU + LCAM (2 CU)   61,554 52.56%  14,018 95.75%  4.39 LUT/sl  +0.018 ns   WORKS
+5 operator IPs (7) 90,047 76.88%  14,634 99.96%  6.15 LUT/sl  -1.549 ns   timing FAIL
+all 8 IPs (10 CU) 124,940  107%      --            --            --       DRC reject
```
**The limit is SLICE OCCUPANCY and its effect on timing, not LUT count.**
At 7 CUs the device still has 23% of its LUTs free, yet 99.96% of slices are
occupied. Vivado packed tighter under pressure exactly as the headroom
suggested -- 4.39 to 6.15 LUT per slice -- and that dense packing is what
destroyed timing: -1.549 ns TNS -7,568 across 21,125 endpoints, plus 251 hold
violations. The spare LUTs were real and unusable.

Detail worth keeping: routing took over two hours and looked like it was
diverging (overlaps 2,020 -> 76,745 -> 204,291). It was not. Each global
iteration begins with a rip-up, so overlaps SPIKE then descend; the run went
on to converge (204,291 -> 3,993) and completed routing. **Do not kill a
congested route on a rising overlap count alone** -- read the trend across a
whole iteration.

At 10 CUs the design does not fit at all, and the reason is the bus, not the
kernels:
```
                     2 CUs                10 CUs
platform (interconnect)   2,861 LUT / 9 BRAM    22,354 LUT / 66 BRAM
```
Attaching 23 extra AXI masters cost **+19,500 LUT and +57 BRAM of pure
interconnect** -- more than the four largest kernels combined, and the BRAM
overrun (312 of 288 RAMB18) is almost entirely platform FIFOs.

**Conclusion: on a KV260 carrying a B4096 DPU at 300 MHz, there is room for
roughly ONE additional custom accelerator, not a nine-operator library.**
That is a real architectural finding and it is the honest answer to "can we
put them all on": no, and here is the number. The deployed
`kv260-hybrid2/3` design -- DPU plus the one operator the DPU cannot express
-- is not a compromise, it is the configuration the device supports.

Not pursued further: a 3-4 CU build would likely close, but each attempt is
3+ hours and the trend between 95.75% and 99.96% slices is steep enough that
the marginal result would not change the conclusion.

---

## 41. *** ONE-COMMAND DEMO: --out MERGES INFERENCE + BOXES, POSTPROCESS COST MEASURED *** (2026-09-12)

Question: `hybrid_pipeline.py --save-npz` then a separate
`postprocess_detections.py --npz ...` call was two commands for one demo
step. Does splitting them cost anything, and can it be one command?

### 41.1 Answer: no FPS cost, because postprocessing was never timed at all
Every FPS figure in this project (§33-§40, `eval_map.py`'s mAP run
included) times ONLY the hardware inference loop -- DPU subgraphs + LCAM
gate + numpy glue. Decode/NMS/drawing/`imwrite` happened in a completely
separate process afterward and was never inside any timing loop. So the
two-command form never cost FPS; it just meant a `.npz` round-trip for a
step that could run in-process.

### 41.2 `postprocess_detections.py` refactored, `hybrid_pipeline.py --out` added
`run_postprocess(pred, scale, px, py, ow, oh, image, out, ...)` was
factored out of the old CLI `main()` so it can be called directly on an
in-memory tensor, not only loaded from a saved `.npz`. `main()` is now a
thin wrapper that loads the `.npz` and calls it -- the old CLI form still
works unchanged, for re-running decode/NMS with different `--conf`/`--nms`
without re-running inference.

`hybrid_pipeline.py` gained `--out <path>`: after the timing loop (and
after the FPS block is printed), it imports `postprocess_detections` and
calls `run_postprocess()` on `out_final` directly. One command, image in,
annotated image out. `pipelined_throughput.py` gained the same flag,
decoding one representative result after its timed loop (every worker
processes the identical preloaded frame, so any one result stands for all
of them).

### 41.3 *** MEASURED: postprocessing costs ~65-68 ms, real and not folded into FPS ***
```
--out on hybrid_pipeline.py (single-frame path):
  decode 15.48 + nms/draw 26.26 + imwrite 25.74 = 67.53 ms

--out on pipelined_throughput.py --workers 4 (after the throughput loop):
  decode 15.20 + nms/draw 24.44 + imwrite 25.01 = 64.70 ms
```
This is comparable in magnitude to the ~76 ms hardware frame itself --
**not negligible**, and worth stating plainly rather than waving away.
Most of it is `imwrite` (JPEG encoding on the ARM core) plus first-call
overhead in `cv2.dnn.NMSBoxes`/grid construction, since this is a single
one-off call, not averaged over repeats the way the hardware loop is.

---

## 42. *** REAL VIDEO THROUGHPUT: 8.93 FPS CAMERA-TO-BOXES, IDENTICAL DETECTIONS HW vs SW *** (2026-09-12)

The first genuine "camera-to-screen" measurement in this project. Every
FPS figure before this one (13.2, 18.7, 19.4, the mAP run) reused ONE
preloaded, pre-quantised tensor and never paid real per-frame video
decode/preprocess/postprocess cost -- true by design (isolating hardware
performance), but never yet measured end to end on video, and never on
video of the actual fire/smoke domain.

### 42.1 *** THE EXISTING "WILDFIRE VIDEOS" WERE FAKE -- CAUGHT BY THE USER, NOT ME ***
Two files already on the board, `wildfire_720p.avi` and
`wildfire_pos_720p.avi` (120 frames each, 1280x720, from an earlier,
undocumented session alongside the Track B 4K work), looked like real
footage from their names and even passed an initial `cv2` probe (sensible
mean pixel values, `read_ok=True`). **The user correctly doubted this** --
a 12-frame contact sheet spanning each ENTIRE clip showed every single
sampled frame identical: same clouds, same parked cars, same fixed
timestamp overlay (`30.Aug 2019 17:04:33` / `17.Dec 2020 07:06:47`) in
every frame. Both are one static surveillance still looped, not video, and
neither shows any fire or smoke. **Lesson: a plausible filename and a
sane-looking pixel mean are not proof of content -- sample frames across
the WHOLE clip and look, especially when a claim is being relied on.**

Two other pre-existing scripts, `lcam_video_4k.py` and `tiled_lcam.py`
(also undocumented, also alongside the Track B work), were checked and
found to default to `XLNX_VART_FIRMWARE=kv260-benchmark-b4096` -- the
STOCK bitstream, driven through `vitis_ai_library.GraphRunner` -- i.e. the
slow CPU-fallback path this whole project exists to fix, not the hybrid
accelerator. Neither was reused.

### 42.2 A real fire/smoke video, built from what genuinely exists
No genuine fire/smoke video existed anywhere in the project. Built one
(`build_wildfire_video.py`) from the SAME 400 labelled photographs already
used for the mAP=0.7360 result (`valset/images/` + `valset/labels/`) --
real, varied, distinct fire/smoke content with ground truth, not synthetic
frames. Source images span >90 distinct native resolutions (240x320 up to
1920x1080); each is letterboxed (grey-padded, matching the model's own
convention) into a fixed 1280x720 canvas -- the single most common native
size in the dataset (186/400) AND the model's training resolution
(4K_VIDEO_RESEARCH.md section 12A) -- so no distortion and no unnecessary
resampling for the majority of frames. A frame-index -> source-filename
manifest is written alongside, so ground truth can be looked up per frame
later.

**Verification gotcha, worth keeping:** the first attempt to spot-check the
new video with `cv2.VideoCapture.set(cv2.CAP_PROP_POS_FRAMES, idx)`
produced a contact sheet that wrongly looked identical to the OLD fake
clip. This board's OpenCV uses a GStreamer backend that had already logged
`Cannot query video position` warnings elsewhere in this session --
`.set(POS_FRAMES)` seeking is unreliable on it and was silently returning
frame 0 regardless of the requested index. Reading SEQUENTIALLY (plain
`cap.read()` in a loop, no `.set()`) showed the correct, varied content,
confirmed identical to reading the source JPGs directly. **Never trust
`.set(CAP_PROP_POS_FRAMES)` on this board's OpenCV build -- read
sequentially.** `video_throughput_hybrid.py` already does this by
construction, so it was never affected.

### 42.3 `video_throughput_hybrid.py` -- the hybrid engine, on real video
Same hand-written engine as `hybrid_pipeline.py` / `pipelined_throughput.py`
(DPU subgraphs on the DPU, the four gates on the LCAM accelerator, LUT/int8
fast paths for the rest), now driving `cv2.VideoCapture` frame by frame
instead of one static tensor. Every stage timed separately, per the
project's convention of never folding an unmeasured cost into a headline
number. Refuses to run if `xdputil query`'s fingerprint is not
`0x101000012010407` -- specifically because `lcam_video_4k.py` sits right
next to it and defaults to the wrong bitstream (section 42.1).

The concat order used by decode is fixed ONCE before the loop rather than
searched every frame (as the one-shot `--out` demo path does): re-deriving
it per frame would be pure waste on a model whose output geometry never
changes, and was exactly the 15 ms/frame the single-image `--out` path
pays for a call that only needs to run once (section 41.3).

### 42.4 *** MEASURED: 400 real frames, hardware gates vs software gates ***
```
                          decode  pre    DPU    gate   CPU   post   TOTAL      FPS
hardware (ip) gates        1.8   25.8   43.0   27.4    2.7   7.3   112.03 ms  8.93
software (numpy) gates     2.0   25.9   43.2    0.0  245.6   7.3   326.60 ms  3.06
```
**454 detections, 196/400 frames flagged, in BOTH runs -- identical to the
frame.** Two different code paths (custom silicon vs software), 400 real
distinct video frames, byte-for-byte the same outcome. This is a stronger
practical correctness result than the earlier `ab_compare.py` single-tensor
diff (section 35.1): it holds across the full diversity of the dataset,
not one demo image.

**Speed-up on real video: 2.92x** (326.60 / 112.03). Lower than the 5.6x
quoted everywhere else in this project, and that difference is itself the
finding: the compute-only figures (75.8 ms hardware / 426.5 ms baseline)
never included the ~34 ms/frame of real decode+preprocess+postprocess that
a camera-fed deployment actually pays. Both numbers are correct; they
answer different questions, and this section exists so neither is quoted
as the other.

**Sanity check against ground truth:** 196/400 frames flagged vs 204/400
frames genuinely containing an object in the valset labels -- close, and
in the expected direction (a plausibility check, not a formal recall
figure; the rigorous mAP=0.7360 result already exists in section 38.1 on
these same images via `eval_map.py`, at a different confidence threshold
and without the video/letterbox round-trip).

### 42.5 One optimisation applied and verified, one identified and left alone
`preprocess` (letterbox + int8 quantisation) measured 32.12 ms/frame on the
first run -- a genuinely NEW cost, never appearing in any prior benchmark
because they all quantised one image once, outside their timing loop.
Following this project's standing rule (every stage gets the same
scrutiny), the quantisation step -- `clip(round(pixel * 2**fp))`, where
`fp` is a fixed xmodel property, not a per-frame value -- was replaced with
a precomputed 256-entry lookup table, verified bit-exact against the
original formula by direct comparison on all 256 possible byte values
before use. Result: 32.12 -> 25.79 ms/frame. **Smaller than expected**: the
bulk of the cost turns out to be `cv2.resize`/letterbox itself, not the
quantisation math, which is now near-free. The resize cost was NOT chased
further -- arbitrary source resolutions must be resized into 640x640
somehow, so this is close to an irreducible floor, and is reported rather
than assumed away.

### 42.6 Artifacts
```
wildfire_real_720p.avi          400 frames, 1280x720, MJPEG, from real photos
wildfire_real_720p.json         frame -> source-image manifest
wildfire_real_720p_det.avi      annotated output, 454 boxes drawn, 115.78 ms/frame
                                 (draw+encode via VideoWriter: 2.10 ms/frame --
                                 much cheaper than the 25.74 ms cv2.imwrite() cost
                                 measured for the single-image --out path, section
                                 41.3, because streaming MJPEG encode avoids
                                 per-call file-open overhead)
```
Pulled to the project at `demo_output/wildfire_real_720p_det.avi` +
`demo_output/wildfire_real_720p_manifest.json`.

### 42.7bis *** WHY DOES IT CALL SMOKE "FIRE"? -- measured, not guessed ***
Raised after watching the annotated video: a distant smoke plume labelled
`fire 0.79`. Is this a hardware/accelerator bug? No -- ruled out by every
correctness result already established (sections 35.1, 36.4, 42.4: the
accelerator is bit-exact against software on this exact model, including
across all 400 frames of this exact video). If the label is wrong, the
mistake is in the model's learned weights, not the custom silicon.

`eval_map.py` gained `--confusion`: it matches every detection against the
best-IoU ground-truth box in its image REGARDLESS of class (the ordinary
per-class AP computation cannot show this -- a wrong-class detection just
looks like one false positive for its own class and one missed detection
for the true class, with no visible link between the two). Run at
`--conf 0.30` to match what the demo video itself uses:
```
fire  -> fire    152   CORRECT
fire  -> smoke     2   *** CONFUSED ***
smoke -> smoke   208   CORRECT
-------------------------------------------------
correctly classified                    : 360
CLASS CONFUSED (right box, wrong label) :   2   (0.6% of matched detections)
no matching ground truth at all         : 104   (a separate, larger problem)
```
**Only 0.6% of detections that landed on a real object had the wrong
label.** The dominant source of imperfect mAP is 104 hallucinations
(detections with no matching object anywhere in the image), not
fire/smoke confusion -- a different problem with a different fix (score
threshold / NMS tuning), not evidence the two classes are poorly
separated.

**The two confusion cases that DO exist were pulled and inspected
directly.** Both (`WEB03809`, conf=0.848, IoU=0.927; `PublicDataset01011`,
conf=0.562, IoU=0.727) are images of UNMISTAKABLE, blazing, fully-involved
fire -- a burning building with the flame front dominating the frame, and
an office-fire test rig fully alight -- where every ground-truth box in
the file is labelled class 1 (smoke) and none is labelled fire. **The
dataset's own annotation is what calls these "smoke"; the model called
them "fire," which is the visually defensible answer.** This is a labelling
inconsistency in the training data, not a demonstrated model weakness --
found by inspecting the actual two failing cases rather than asserting a
cause.

**For a specific ambiguous single frame (a distant plume with a small
visible glow at its base, moderate confidence ~0.79)**, no source-image
match was pinned down from a screenshot alone, so no specific claim is
made about that one frame beyond what the aggregate measurement supports:
fire/smoke misclassification is real but rare (0.6%), and where it occurs
in this dataset it is at least partly a property of inconsistent ground-
truth labelling rather than a proven model deficiency. **For the journal:
report the 0.6% figure and the two labelling-inconsistency examples
together** -- both the model's real limitation and the ground truth's
real limitation are part of an honest accuracy discussion, and dataset
label noise on a scraped multi-source wildfire dataset (`AoF*`, `WEB*`,
`PublicDataset*` prefixes visible in the filenames -- several distinct
sources, not one consistent annotation effort) is itself a defensible,
citable methodological observation.

### 42.7 Standing numbers, all scopes stated explicitly
```
hardware-only, static tensor reused    :  75.8 ms  -> 13.2 FPS  (sections 33-40)
hardware-only, pipelined 4 workers     :  51.7 ms  -> 19.3 FPS  (sections 38, 41)
REAL VIDEO, camera to boxes, hybrid    : 112.0 ms  ->  8.9 FPS  (this section)
REAL VIDEO, camera to boxes, software  : 326.6 ms  ->  3.1 FPS  (this section)
  speed-up on real video               :   2.92x
  detections: 454/400 frames, IDENTICAL in both video runs
```

Both `--out` calls reproduced the IDENTICAL three detections already
verified in §35.3 (`fire 0.875 [378,5,769,477]`,
`smoke 0.651 [469,418,713,494]`, `smoke 0.301 [607,513,634,541]`),
confirming the refactor changed no behaviour.

### 41.4 Reproduced: 19.3 FPS at 4 workers
```
python3 pipelined_throughput.py --workers 4 --frames 40
  throughput  :  51.73 ms/frame -> 19.33 FPS   (was 19.43 FPS, §38.3 -- within noise)
  latency     : 206.32 ms mean
```
Confirms §38.3's 4-worker figure is reproducible, not a one-off.

### 41.5 Honest note for the write-up
None of the project's FPS/throughput claims (13.2, 18.7, 19.4 FPS, the
mAP run) include image decode, letterbox, quantisation, or postprocessing
-- they are hardware-inference-only, consistently, throughout. That is a
defensible and clearly-stated scope (matches how DPU-only benchmarks are
conventionally reported), but a true "camera frame in, annotated frame
out" wall-clock number would need to add roughly **~65-70 ms** of
software pre/postprocessing on top of the ~76 ms hardware figure -- i.e.
close to halving the naive top-line FPS if ever quoted as an end-to-end
"screen-to-screen" number. State the scope explicitly in the journal
rather than letting a reader assume the FPS already includes it.
