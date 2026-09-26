# sweep_bram.tcl -- how much BRAM does the LCAM kernel actually need?
#
# The kernel's 7 placed BRAMs are not compute storage: csynth reports
# BRAM_18K = 0 for the datapath. They are the m_axi adapter FIFOs, whose
# depth is roughly num_outstanding x max_burst_length. At the current
# 16 x 64 that is 1024 entries x 128 bits = 16 KB per port, which synthesis
# maps into block RAM. Three ports gives about 7 tiles.
#
# Smaller FIFOs should fall back to LUTRAM/SRL and free the BRAM. The risk is
# throughput: fewer outstanding reads means less latency hiding on the AXI
# port. Worth checking cheaply here rather than after a 1-hour link.
#
# Run:  vitis_hls -f sweep_bram.tcl

set variants {
    {16 64  baseline}
    { 8 32  half}
    { 4 32  quarter}
    { 4 16  minimal}
}

foreach v $variants {
    set nout  [lindex $v 0]
    set burst [lindex $v 1]
    set name  [lindex $v 2]

    # regenerate the source with the FIFO parameters substituted
    set fin  [open src/lcam_attention_gate_opt.cpp r]
    set body [read $fin]
    close $fin
    regsub -all {num_read_outstanding=16}  $body "num_read_outstanding=$nout"  body
    regsub -all {num_write_outstanding=16} $body "num_write_outstanding=$nout" body
    regsub -all {max_read_burst_length=64}  $body "max_read_burst_length=$burst"  body
    regsub -all {max_write_burst_length=64} $body "max_write_burst_length=$burst" body
    file mkdir sweep
    set fout [open sweep/v_$name.cpp w]
    puts $fout $body
    close $fout

    open_project -reset sweep_$name
    set_top lcam_attention_gate_opt
    add_files sweep/v_$name.cpp -cflags "-Isrc"
    open_solution -reset -flow_target vitis sol1
    set_part {xck26-sfvc784-2LV-c}
    create_clock -period 3.333 -name default
    config_interface -m_axi_max_widen_bitwidth 128
    csynth_design
    puts "SWEEP_RESULT $name outstanding=$nout burst=$burst"
}
puts "SWEEP DONE"
exit
