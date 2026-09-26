# ============================================================================
# rdata_recapture.tcl -- Re-run the §19 RDATA capture on the CURRENT bitstream
# ============================================================================
# Goal (per PROJECT_HISTORY.md §19 + PROJECT_CONTEXT.md "Suggested next steps" #2):
# re-verify whether the "random-looking" RDATA pattern on head_transpose's
# read channel still appears, and this time check EVERY beat where
# RVALID & RREADY are both 1 across the FULL 1024-sample window, not just
# the first one (last session only inspected the first transfer beat).
#
# Run INTERACTIVELY: & "C:\Xilinx\Vivado\2022.2\bin\vivado.bat" -mode tcl
# Paste each phase as ONE line (multi-line paste into the Tcl console has
# corrupted for-loops before -- see vivado_scripts/README.md gotcha #6).
#
# IMPORTANT: the board's PL is currently running the STOCK kv260-benchmark-
# b4096 app (confirmed via `xmutil listapps` on 2026-08-28) -- our design
# is NOT currently on the chip. JTAG programming below overwrites whatever
# is live; it does NOT touch flash/QSPI/SD, so it's volatile and harmless,
# but it DOES mean xmutil's bookkeeping and the actual FPGA contents will
# disagree until the board is next power-cycled or something reloads via
# xmutil again. Fine for a debug session; just don't forget it's happened.
# ============================================================================

# ── PHASE 0: connect and check what's already built ─────────────────────────
open_project C:/Xilinx/projects/kv260_custom_noDPU/KV260.xpr
# MUST be open_run synth_1, NOT impl_1_01 -- debug-port editing
# (create_debug_port/connect_debug_port, used in PHASE 2) only works on
# the pre-placement synthesized netlist. Running it against an implemented
# (placed+routed) design fails with "[Vivado 12-4097] This Vivado Debug
# command cannot be performed on the current design" (hit this live on
# 2026-08-28 -- see vivado_scripts/README.md gotcha #7).
open_run synth_1
# Check if a debug core (from the §17-19 sessions) is already present in
# this run's netlist -- if last session's u_ila_0 + RDATA probes survived
# on synth_1, phases 1-2 can be skipped entirely and you can jump straight
# to PHASE 4 (program + arm). Don't assume this matches whatever an
# impl_1_01 check may have shown -- synth_1 and an implemented run of it
# are separate netlist states; re-check fresh here.
puts "EXISTING DEBUG CORES: [get_debug_cores -quiet]"
puts "EXISTING DEBUG PORTS on u_ila_0 (if any): [get_debug_ports -of_objects [get_debug_cores u_ila_0] -quiet]"
# If the output above already lists probes named *RDATA*, *RVALID*, *RREADY*
# -- STOP, skip to PHASE 4. Otherwise continue to PHASE 1.
#
# Also record how many probes already exist, if the core exists at all --
# needed in PHASE 2 to know which probe index to start creating NEW ones
# at (create_debug_port auto-numbers sequentially from whatever's already
# there; get this wrong and you silently overwrite/misconnect an existing
# probe instead of adding a new one).
#
# NOTE: get_debug_ports on an existing core returns clk PLUS every probeN
# -- if you count that whole list, you're off by one (found this the hard
# way: a real core came back with clk+probe0..probe5 = 7 entries, but only
# 6 are actual probes; the next auto-created one would be probe6, not
# probe7). Filter clk out explicitly:
set all_ports [get_debug_ports -of_objects [get_debug_cores u_ila_0 -quiet] -quiet]
set probe_ports [lsearch -all -inline -glob $all_ports "*probe*"]
set EXISTING_PROBE_COUNT [llength $probe_ports]
puts "EXISTING PROBE COUNT (0 if u_ila_0 doesn't exist yet; clk excluded): $EXISTING_PROBE_COUNT"
# If EXISTING_PROBE_COUNT > 0, also check what's already there before
# adding more -- don't assume it's from §18-19 just because a core exists:
foreach p $probe_ports { puts "$p -> [get_nets -of_objects [get_debug_ports $p]]" }

# ── PHASE 1: does a parent vector net even exist? TEST BEFORE MARKING ───────
# We already got burned by this exact shortcut for AWADDR/ARADDR: post-
# synthesis, Vivado had split the bus into 128 separate single-bit nets --
# no parent vector net existed, so a single bulk mark_debug/get_nets call
# on the whole bus silently resolved to nothing. Test for the SAME failure
# mode on RDATA before trying the bulk form again:
puts "PARENT VECTOR NET TEST (expect EMPTY if it's already split, like AWADDR/ARADDR were): [get_nets top_i/smartconnect_hp2/M00_AXI_rdata -quiet]"
puts "PER-BIT TEST (bit 0 only): [get_nets -of_objects [get_pins {top_i/smartconnect_hp2/M00_AXI_rdata[0]}] -quiet]"
# Confirm the actual bit width too -- §19 used 128 (HP-port width):
puts "RDATA WIDTH: [llength [get_pins {top_i/smartconnect_hp2/M00_AXI_rdata[*]}]]"
puts "RVALID net: [get_nets -of_objects [get_pins top_i/smartconnect_hp2/M00_AXI_rvalid]]"
puts "RREADY net: [get_nets -of_objects [get_pins top_i/smartconnect_hp2/M00_AXI_rready]]"
#
# DECISION POINT based on the two lines above:
#   - If it came back EMPTY (most likely, matching AWADDR/ARADDR's
#     behavior) -- the bus is already split into single-bit nets, and
#     PHASE 2 STEP B's per-bit create_debug_port/connect_debug_port loop
#     below is required as written.
#   - If the PARENT VECTOR NET TEST unexpectedly returned something real
#     (non-empty) -- the per-bit loop below still works correctly either
#     way, it's just doing 128 small steps instead of 1. Not worth
#     special-casing; only worth noting if you want to save build-graph
#     size by connecting one 128-bit probe to the whole vector instead.

# ── PHASE 2, STEP A: bootstrap u_ila_0 IF IT DOESN'T EXIST YET ───────────────
# This is the CONFIRMED verbatim sequence actually used to bootstrap the
# core the first time in §17 (independently verified -- not a guess, unlike
# the mark_debug+auto-implement flow this replaces).
#
# *** ONLY RUN THIS BLOCK (lines below down to the connect_debug_port/clk
# line) IF PHASE 0's EXISTING_PROBE_COUNT WAS 0. *** If a u_ila_0 already
# exists from a prior session, SKIP straight to STEP B below -- running
# create_debug_core again here would create a SECOND, conflicting core.
create_debug_core u_ila_0 ila
set_property C_DATA_DEPTH 1024 [get_debug_cores u_ila_0]
set_property C_TRIGIN_EN false [get_debug_cores u_ila_0]
set_property C_TRIGOUT_EN false [get_debug_cores u_ila_0]
set_property C_ADV_TRIGGER false [get_debug_cores u_ila_0]
set_property C_INPUT_PIPE_STAGES 0 [get_debug_cores u_ila_0]
set_property C_EN_STRG_QUAL false [get_debug_cores u_ila_0]
set_property C_CLK_INPUT_FREQ_HZ 100000000 [get_debug_cores u_ila_0]
connect_debug_port u_ila_0/clk [get_nets top_i/zynq_ultra_ps_e/pl_clk0]

# ── PHASE 2, STEP B: create + connect one probe per bit (per-bit, per README gotcha #6) ──
# Works identically whether u_ila_0 was just bootstrapped above (start at
# probe0) or already existed from a prior session (start after whatever's
# already there) -- $EXISTING_PROBE_COUNT from PHASE 0 covers both cases.
# SINGLE-LINE for-loop with semicolons -- multi-line paste has corrupted
# this before (loop ran out of range before breaking).
# NOTE: create_debug_port auto-names sequentially (probe0, probe1, ...) --
# the loop assumes that numbering exactly matches $EXISTING_PROBE_COUNT +
# loop position. If PHASE 0 showed any gaps/unusual names in the existing
# probe list, verify with get_debug_ports after creating just the first
# one before trusting the rest of the loop.
for {set i 0} {$i < 128} {incr i} { set probe_num [expr {$EXISTING_PROBE_COUNT + $i}]; create_debug_port u_ila_0 probe; connect_debug_port u_ila_0/probe$probe_num [get_nets -of_objects [get_pins "top_i/smartconnect_hp2/M00_AXI_rdata[$i]"]] }
set RVALID_PROBE [expr {$EXISTING_PROBE_COUNT + 128}]
create_debug_port u_ila_0 probe
connect_debug_port u_ila_0/probe$RVALID_PROBE [get_nets -of_objects [get_pins top_i/smartconnect_hp2/M00_AXI_rvalid]]
set RREADY_PROBE [expr {$EXISTING_PROBE_COUNT + 129}]
create_debug_port u_ila_0 probe
connect_debug_port u_ila_0/probe$RREADY_PROBE [get_nets -of_objects [get_pins top_i/smartconnect_hp2/M00_AXI_rready]]
puts "RDATA probes: $EXISTING_PROBE_COUNT..[expr {$EXISTING_PROBE_COUNT + 127}] / RVALID probe$RVALID_PROBE / RREADY probe$RREADY_PROBE"

# Sanity-check the loop actually connected all 130 new probes before
# proceeding -- don't just trust it ran without error (that's exactly how
# the AWADDR/ARADDR bulk-mark failure went unnoticed until CSV analysis):
puts "TOTAL PROBE COUNT NOW (expect EXISTING_PROBE_COUNT + 130): [llength [get_debug_ports -of_objects [get_debug_cores u_ila_0]]]"
# (Keep ARVALID/ARADDR probed too if still present from §18 -- useful to
# confirm the read address handshake still lines up the same way. If they
# need re-adding, use the same create_debug_port+connect_debug_port
# per-bit pattern above, continuing the probe numbering from wherever
# this block left off.)

# ── PHASE 3: rebuild with debug cores inserted ───────────────────────────────
# This re-synthesizes the debug netlist and re-implements+re-bitstreams.
# ~20-25 min per README.md's own note on prior debug-probe cycles.
implement_debug_core [get_debug_cores]
write_checkpoint -force C:/Xilinx/rdata_recapture.dcp
opt_design
place_design
route_design
write_debug_probes -force C:/Xilinx/probes_rdata.ltx
write_bitstream -force C:/Xilinx/rdata_recapture.bit
puts "REBUILD STATUS: bitstream written -- proceed to PHASE 4"

# ── PHASE 4: JTAG program + arm trigger ──────────────────────────────────────
open_hw_manager
connect_hw_server
open_hw_target
current_hw_device [get_hw_devices xck26_0]
set_property PROGRAM.FILE {C:/Xilinx/rdata_recapture.bit} [current_hw_device]
program_hw_devices [current_hw_device]
refresh_hw_device [current_hw_device]
set_property PROBES.FILE {C:/Xilinx/probes_rdata.ltx} [current_hw_device]
refresh_hw_device [current_hw_device]
puts "AVAILABLE PROBES: [get_hw_probes -of_objects [get_hw_ilas]]"

# Trigger on RVALID (data-transfer beat) -- use the EXACT name PHASE 4's
# probe listing prints (expect the legacy hp0 label per §19's gotcha,
# ALL-CAPS for the wide signals, e.g. top_i/smartconnect_hp0_M00_AXI_RVALID).
# Fill in the real name before running the next two lines:
set_property TRIGGER_COMPARE_VALUE eq1'b1 [get_hw_probes <PASTE_REAL_RVALID_PROBE_NAME_HERE>]
run_hw_ila [get_hw_ilas]

# ── PHASE 5: run the test on the board, THEN come back and: ─────────────────
#   ssh in and run: python3 test_head_transpose_debug.py
#   (prints in_buf.phys_addr/out_buf.phys_addr inline -- needed to compare
#   against captured ARADDR from the SAME run, per §18's methodology note:
#   never compare against a phys_addr from a different script invocation)
wait_on_hw_ila [get_hw_ilas]
puts "CORE STATUS (look for CORE_STATUS = FULL, not WAITING FOR TRIGGER):"
report_property [get_hw_ilas]
upload_hw_ila_data [get_hw_ilas]
set ila_data [current_hw_ila_data]
write_hw_ila_data -force -csv_file C:/Xilinx/rdata_recapture.csv $ila_data
puts "CSV written to C:/Xilinx/rdata_recapture.csv -- for full-window analysis"

# ── PHASE 6: analysis to do on the CSV once it's back ────────────────────────
# Unlike §19 (which only checked the FIRST RVALID&RREADY beat), scan ALL
# 1024 samples for every beat where RVALID=1 AND RREADY=1, assemble each
# beat's 128-bit RDATA (per-bit loop, not manual eyeballing -- README
# gotcha), and compare EACH ONE against the expected input bytes
# (src = np.arange(48) in test_head_transpose_debug.py) at the
# corresponding offset. This checks whether the "random-looking" pattern
# from §19 was a one-off (e.g. genuinely mid-transition data) or is
# consistent across the whole burst -- that distinction matters for
# whether this points at a timing race vs. a structural DRAM/addressing
# problem.
