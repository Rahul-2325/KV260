#*****************************************************************
# run_hls_opt.tcl -- synthesise the OPTIMISED lcam_attention_gate
#
# Produces an IP with a 512-bit AXI datapath and pure-integer
# arithmetic, replacing the original 8-bit/float version.
#
# The arithmetic was proven BIT-IDENTICAL to the original across the
# full int8 x int8 domain for all four real layer configs -- see
# debug_scripts/verify_lcam_int_math.py.
#
# Run:
#   cd <this directory>
#   & "C:\Xilinx\Vitis_HLS\2022.2\bin\vitis_hls.bat" -f run_hls_opt.tcl
#
# Output IP zip lands in:
#   lcam_opt_prj/sol1/impl/export.zip
# which then gets unpacked into C:\Xilinx\projects\custom_ip_repo\
#*****************************************************************

open_project -reset lcam_opt_prj
set_top lcam_attention_gate_opt
add_files lcam_attention_gate_opt.cpp -cflags "-I."

open_solution -reset "sol1" -flow_target vivado
# KV260 part, matching the rest of the project
set_part {xck26-sfvc784-2LV-c}
# 100 MHz -- the IP sits on pl_clk0 like every other custom IP here
create_clock -period 10 -name default

# report the interface/latency numbers we care about for the paper
csynth_design

# package as a Vivado IP so it can be dropped into custom_ip_repo
export_design -format ip_catalog -rtl verilog \
    -display_name "LCAM Attention Gate (optimised)" \
    -description "int8 attention gate, 512-bit AXI datapath, integer arithmetic" \
    -vendor "xilinx.com" -library "hls" -version "2.0"

puts "=========================================="
puts "HLS DONE"
puts "  synthesis report : lcam_opt_prj/sol1/syn/report/lcam_attention_gate_opt_csynth.rpt"
puts "  exported IP      : lcam_opt_prj/sol1/impl/export.zip"
puts "=========================================="
exit
