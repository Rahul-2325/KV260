# Vivado Scripts

**These are RECONSTRUCTED** from the exact commands documented as working in
`PROJECT_HISTORY.md`. The originals lived on the Windows machine under
`C:\Xilinx\` and were not part of this package. They match the documented
working commands but have not been re-run from these exact files.

- `rebuild_synth.tcl` — synthesis (~15-20 min, needs ~10.5GB RAM free)
- `rebuild_impl.tcl` — implementation + bitstream (~20 min)
- `jtag_program_and_ila.tcl` — JTAG program + ILA capture (interactive)

## Hard-won gotchas (all cost real time to discover)

1. **Always use `-to_step write_bitstream`** on impl, or it silently stops
   before producing a bitstream.
2. **Never interrupt a run.** Partial progress is not resumable.
3. **Free RAM before synthesis** — it peaks around 10.5GB and has crashed
   the machine mid-run before.
4. **Net names are NOT trustworthy.** Repeated rewiring left stale labels
   (a net physically on `smartconnect_hp2` may be named `smartconnect_hp0_*`).
   Always resolve by pin:
   `get_bd_intf_nets -of_objects [get_bd_intf_pins <cell>/<port>]`
5. **`validate_bd_design` fails harmlessly** with a `.hpfm`/platform-clock
   error (Windows path-with-spaces issue). It has never blocked a real
   build — proceed past it.
6. **Paste multi-line Tcl loops as ONE line** with semicolons. Multi-line
   paste into the Vivado console gets mangled and can run past array bounds.
7. **Debug-port editing (`create_debug_port`/`connect_debug_port`) requires
   `open_run synth_1`, NOT `open_run impl_1_01`.** Running these against an
   implemented (placed+routed) design fails with `[Vivado 12-4097] This
   Vivado Debug command cannot be performed on the current design` --
   close the design and reopen the synthesized run first. Also don't
   assume debug-core state (e.g. probe count) carries over identically
   between `synth_1` and an implemented run of it -- re-check fresh on
   whichever one you actually need to edit; they're separate in-memory
   netlist states and this project has already seen one case (§16) where
   a debug-core insertion didn't persist across a session.
8. **`get_debug_ports` counts `u_ila_0/clk` as a port**, alongside every
   `probeN`. Naively `llength`-ing that list is off-by-one when computing
   the next auto-assigned probe index. Filter first:
   `lsearch -all -inline -glob $all_ports "*probe*"`.
9. **DO NOT JTAG-program a bitstream and then poke the IPs from Linux.**
   `program_hw_devices` only rewrites the PL fabric; it does NOT update
   the device tree, clocks/resets, or zocl's IP-layout registration the
   way `xmutil loadapp` does. Any AXI-Lite register access afterwards can
   hit an address with no responding slave, which hangs the ZynqMP
   interconnect and reboots the board. This cost 5 board crashes in one
   session before it was identified (PROJECT_HISTORY.md §25). For ILA
   work, package the debug design as its own xmutil app instead.
10. **Always verify which bitstream you actually programmed.** After
    `program_hw_devices`, run
    `puts "ILA CORES: [llength [get_hw_ilas -quiet]]"` -- and read the
    probes-file mismatch warnings. `impl_1_01/top_wrapper.bit` turned out
    to contain an OLD debug core, not the clean non-debug build it was
    assumed to be; testing against it unverified would have silently
    invalidated a whole experiment.
11. **`pscp` needs its own `-batch -hostkey ... -scp` flags.** Unlike
    `plink` it doesn't inherit the cached host key (hangs on an
    unanswerable prompt), and this board's minimal image has no SFTP
    server, so without `-scp` it fails with
    `sh: /usr/libexec/sftp-server: No such file or directory`.
