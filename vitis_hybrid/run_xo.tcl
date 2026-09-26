# run_xo.tcl -- Vitis HLS: export lcam_attention_gate_opt as a Vitis KERNEL (.xo)
#
# Difference from run_hls_opt.tcl (which produced a Vivado IP): the solution
# is opened with -flow_target vitis, and export_design uses -format xo.
# Only that combination produces a kernel XRT will drive with the standard
# AP_CTRL_HS handshake -- which is the whole point here, since the raw
# Vivado-IP path is what left XRT writing ap_start into a read-only DPU
# version register (PROJECT_HISTORY.md 29.x).
#
# The source already has the required kernel-shaped interface
# (all scalars s_axilite bundle=control, port=return bundle=control,
# m_axi offset=slave), so no source change is needed.
#
# Clock: 300 MHz = platform clock index 1 (the DPU's own core clock).
# If timing does not close we fall back to clock index 0 (100 MHz).

# NOTE: vitis_hls -f does NOT forward trailing args as $argv (it hands the
# script "-f" as argv 0), so the period comes in through the environment.
set PERIOD 3.333
if {[info exists ::env(LCAM_PERIOD)]} { set PERIOD $::env(LCAM_PERIOD) }

open_project -reset lcam_xo_prj
set_top lcam_attention_gate_opt
add_files src/lcam_attention_gate_opt.cpp -cflags "-Isrc"

open_solution -reset -flow_target vitis sol1
set_part {xck26-sfvc784-2LV-c}
create_clock -period $PERIOD -name default

# -flow_target vitis silently sets m_axi_max_widen_bitwidth=512, which
# would widen our deliberately 128-bit ports straight back to 512 and undo
# the area saving. Pin it to the real S_AXI_HP width.
config_interface -m_axi_max_widen_bitwidth 128

csynth_design
export_design -format xo -output lcam_attention_gate_opt.xo

puts "XO_BUILD_DONE period=$PERIOD"
exit
