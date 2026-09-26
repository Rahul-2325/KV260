#*****************************************************************
# build.tcl -- minimal bitstream carrying ONLY the optimised
#              lcam_attention_gate (512-bit AXI, integer arithmetic).
#
# Purpose: measure the optimised IP on real hardware without waiting on
# a full 9-IP or DPU build. Small design -> fast synthesis.
#
# Addresses are chosen to MATCH board_driver/hw_ip_driver.py's existing
# IP_ADDR entry for lcam_attention_gate, so the current driver and
# benchmark scripts work unchanged:
#     s_axi_control    0x8006_0000
#     s_axi_control_r  0x800F_0000
#
# Data masters go to HP0 (segment covers LOW DDR 0x0-0x7FFF_FFFF), which
# is the only region the PL can reach -- see PROJECT_HISTORY.md ??26. The
# driver's MIN_CMA_ALLOC (>=64KB) keeps allocations in that region.
#*****************************************************************

set PRJ  C:/Xilinx/projects/lcam_opt_bit/prj
set REPO C:/Xilinx/projects/custom_ip_repo_opt
set OUT  C:/Xilinx/projects/lcam_opt_bit/out
file mkdir $OUT

create_project -force lcam_opt_bit $PRJ -part xck26-sfvc784-2LV-c
set_property board_part xilinx.com:kv260_som:part0:1.4 [current_project]
set_property ip_repo_paths $REPO [current_project]
update_ip_catalog -rebuild
puts "IPDEF: [get_ipdefs -all *lcam_attention_gate_opt*]"

create_bd_design "top"

# ---- PS ------------------------------------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.4 ps
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config {apply_board_preset "1"} [get_bd_cells ps]
# NOTE on which GP master to use:
#   GP0 = M_AXI_HPM0_FPD -> apertures start at 0xA000_0000, so it CANNOT
#         reach 0x8006_0000 ("The proposed address must fit an available
#         aperture ... Valid apertures are {0xA000_0000, ...}")
#   GP2 = M_AXI_HPM0_LPD -> this is the one that provides the
#         0x8000_0000 aperture, and is what the DPU design used to place
#         this same IP at 0x8006_0000.
# GP1 (M_AXI_HPM1_FPD) is explicitly disabled; the board preset turns it
# on and then validation fails with "clock pins are not connected to a
# valid clock source: /ps/maxihpm1_fpd_aclk".
set_property -dict [list \
  CONFIG.PSU__USE__M_AXI_GP0 {0} \
  CONFIG.PSU__USE__M_AXI_GP1 {0} \
  CONFIG.PSU__USE__M_AXI_GP2 {1} \
  CONFIG.PSU__MAXIGP2__DATA_WIDTH {32} \
  CONFIG.PSU__USE__S_AXI_GP2 {1} \
  CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
  CONFIG.PSU__FPGA_PL0_ENABLE {1} \
  CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100} \
  CONFIG.PSU__NUM_FABRIC_RESETS {1} \
] [get_bd_cells ps]

# ---- the optimised IP ----------------------------------------------
create_bd_cell -type ip -vlnv xilinx.com:hls:lcam_attention_gate_opt:2.0 lcam

# ---- reset ----------------------------------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rstgen
connect_bd_net [get_bd_pins ps/pl_clk0]    [get_bd_pins rstgen/slowest_sync_clk]
connect_bd_net [get_bd_pins ps/pl_resetn0] [get_bd_pins rstgen/ext_reset_in]

# ---- control path: PS M_AXI_HPM0_FPD -> smartconnect -> 2 slaves ----
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 sc_ctrl
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2}] [get_bd_cells sc_ctrl]
connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_HPM0_LPD] [get_bd_intf_pins sc_ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_ctrl/M00_AXI]   [get_bd_intf_pins lcam/s_axi_control]
connect_bd_intf_net [get_bd_intf_pins sc_ctrl/M01_AXI]   [get_bd_intf_pins lcam/s_axi_control_r]

# ---- data path: 3 masters -> smartconnect -> HP0 --------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 sc_hp
set_property -dict [list CONFIG.NUM_SI {3} CONFIG.NUM_MI {1}] [get_bd_cells sc_hp]
connect_bd_intf_net [get_bd_intf_pins lcam/m_axi_gmem0] [get_bd_intf_pins sc_hp/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins lcam/m_axi_gmem1] [get_bd_intf_pins sc_hp/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins lcam/m_axi_gmem2] [get_bd_intf_pins sc_hp/S02_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_hp/M00_AXI]    [get_bd_intf_pins ps/S_AXI_HP0_FPD]

# ---- clocks / resets -------------------------------------------------
set CLK  [get_bd_pins ps/pl_clk0]
set RSTN [get_bd_pins rstgen/peripheral_aresetn]
foreach p {lcam/ap_clk sc_ctrl/aclk sc_hp/aclk ps/maxihpm0_lpd_aclk ps/saxihp0_fpd_aclk} {
    connect_bd_net $CLK [get_bd_pins $p]
}
foreach p {lcam/ap_rst_n sc_ctrl/aresetn sc_hp/aresetn} {
    connect_bd_net $RSTN [get_bd_pins $p]
}

# ---- addresses ------------------------------------------------------
assign_bd_address
foreach seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces ps/Data]] {
    puts "seg: $seg  [get_property offset [get_bd_addr_segs $seg]]"
}
catch { set_property offset 0x80060000 [get_bd_addr_segs ps/Data/SEG_lcam_Reg]   }
catch { set_property range  64K        [get_bd_addr_segs ps/Data/SEG_lcam_Reg]   }
catch { set_property offset 0x800F0000 [get_bd_addr_segs ps/Data/SEG_lcam_Reg_1] }
catch { set_property range  64K        [get_bd_addr_segs ps/Data/SEG_lcam_Reg_1] }
puts "=== FINAL ADDRESS MAP ==="
foreach seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces ps/Data]] {
    puts [format "  %-40s %s +%s" [file tail $seg] \
        [get_property offset [get_bd_addr_segs $seg]] [get_property range [get_bd_addr_segs $seg]]]
}

validate_bd_design -force
save_bd_design
make_wrapper -files [get_files $PRJ/lcam_opt_bit.srcs/sources_1/bd/top/top.bd] -top -force
add_files -norecurse $PRJ/lcam_opt_bit.gen/sources_1/bd/top/hdl/top_wrapper.v
set_property top top_wrapper [current_fileset]
update_compile_order -fileset sources_1

# ---- build -----------------------------------------------------------
launch_runs synth_1 -jobs 4
wait_on_run synth_1
puts "SYNTH: [get_property STATUS [get_runs synth_1]]"
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} { puts "SYNTH FAILED"; return }

launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
puts "IMPL: [get_property STATUS [get_runs impl_1]]"
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} { puts "IMPL FAILED"; return }

open_run impl_1
puts "WNS: [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]"
puts "WHS: [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -hold]]"
report_utilization -file $OUT/utilization.rpt
report_power       -file $OUT/power.rpt

set bit [glob -nocomplain $PRJ/lcam_opt_bit.runs/impl_1/*.bit]
if {[llength $bit] > 0} { file copy -force [lindex $bit 0] $OUT/lcam_opt.bit; puts "BIT: $OUT/lcam_opt.bit" }
write_hw_platform -fixed -include_bit -force $OUT/lcam_opt.xsa
puts "BUILD DONE"

