#*****************************************************************
# run_hls_opt.tcl -- synthesise the OPTIMISED lcam_attention_gate
#
# NOTE: this must live under a path with NO SPACES. Vitis HLS refuses:
#   ERROR: [HLS 200-70] Project/solution path '...Punnam Rahul...'
#          contains illegal character ' '.
# Hence C:\Xilinx\hls_work\lcam_opt rather than the project folder.
#
# The arithmetic was proven BIT-IDENTICAL to the original float version
# across the full int8 x int8 domain for all four real layer configs
# (debug_scripts/verify_lcam_int_math.py).
#*****************************************************************

open_project -reset lcam_opt_prj
set_top lcam_attention_gate_opt
add_files lcam_attention_gate_opt.cpp -cflags "-I."

open_solution -reset "sol1" -flow_target vivado
set_part {xck26-sfvc784-2LV-c}
create_clock -period 10 -name default

csynth_design

export_design -format ip_catalog -rtl verilog \
    -display_name "LCAM Attention Gate optimised" \
    -description "int8 attention gate, 512-bit AXI datapath, integer arithmetic" \
    -vendor "xilinx.com" -library "hls" -version "2.0"

puts "=========================================="
puts "HLS DONE"
puts "  report : lcam_opt_prj/sol1/syn/report/lcam_attention_gate_opt_csynth.rpt"
puts "  IP zip : lcam_opt_prj/sol1/impl/export.zip"
puts "=========================================="
exit
