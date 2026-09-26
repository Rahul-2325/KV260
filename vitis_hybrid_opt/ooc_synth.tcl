# ooc_synth.tcl -- out-of-context synthesis of the generated LCAM RTL.
#
# csynth reports BRAM_18K = 0 for every FIFO variant because it does not model
# the m_axi adapter buffers in that table -- yet the placed design shows 7 BRAM
# inside this kernel. Only real synthesis resolves the difference, and OOC
# synthesis of just this module takes minutes instead of the hour a full
# v++ link would cost.
foreach v {baseline minimal} {
    set dir C:/Xilinx/hls_work/lcam_xo/sweep_$v/sol1/syn/verilog
    if {![file isdirectory $dir]} { puts "MISSING $dir"; continue }
    create_project -in_memory -part xck26-sfvc784-2LV-c
    add_files [glob $dir/*.v]
    set_property top lcam_attention_gate_opt [current_fileset]
    synth_design -top lcam_attention_gate_opt -part xck26-sfvc784-2LV-c \
                 -mode out_of_context
    puts "===== OOC RESULT $v ====="
    report_utilization -file C:/Xilinx/hls_work/lcam_xo/ooc_$v.rpt
    close_project
}
puts "OOC DONE"
exit
