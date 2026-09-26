#*****************************************************************
# build_platform.tcl -- minimal EXTENSIBLE (accelerated) KV260 platform
#
# WHY THIS EXISTS
# ---------------
# Previous v++ attempts failed with:
#     ERROR: [v++ 60-1606] Platform 'kv260_lcam.xsa' is a non-accelerated
#            platform. By policy, ...
#     ERROR: [v++ 60-2500] Valid input file of type XCLBIN is required
# An XSA exported from an ordinary Vivado design is a FIXED platform.
# v++ can only link kernels into an EXTENSIBLE platform, i.e. one whose
# block design publishes PFM_NAME / PFM.CLOCK / PFM.AXI_PORT / PFM.IRQ
# and sets platform.extensible = true.
#
# `platforminfo` on the existing kv260_9ip_v2.xpfm confirmed the problem:
# it lists NO clocks and NO AXI ports.
#
# This script builds such a platform for KV260 from scratch. It is
# modelled on the DPU-TRD's xilinx_zcu102_base pfm_decls.tcl, but written
# minimally rather than adapting that platform's 89 KB ZCU102-specific
# dr.bd.tcl (different board, different PS config and IO).
#
# The platform deliberately contains NO DPU and NO custom IP -- v++ will
# instantiate dpu.xo and lcam_attention_gate_opt.xo as KERNELS into it,
# generating the bitstream and a matching xclbin. That is what makes the
# DPU XRT-dispatchable, which the Vivado-TRD flow never could (§31).
#
# Clocks published: 100 MHz (default), 300 MHz, 600 MHz
#   -- the DPU-TRD prj_config asks for
#        freqHz=300000000:DPUCZDX8G_1.aclk
#        freqHz=600000000:DPUCZDX8G_1.ap_clk_2
#*****************************************************************

set PRJ  C:/Xilinx/projects/kv260_ext_pfm/prj
set OUT  C:/Xilinx/projects/kv260_ext_pfm/out
set PNAME kv260_ext
set VER   1.0
file mkdir $OUT

create_project -force $PNAME $PRJ -part xck26-sfvc784-2LV-c
set_property board_part xilinx.com:kv260_som:part0:1.4 [current_project]

create_bd_design "top"

# ---------------- PS ------------------------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.4 ps_e
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config {apply_board_preset "1"} [get_bd_cells ps_e]

# Enable the masters/slaves the platform will publish to kernels.
#   GP0 = M_AXI_HPM0_FPD  (control path for kernels)
#   GP2 = M_AXI_HPM0_LPD  (not published; keeps 0x8000_0000 free)
#   S_AXI_GP2..GP5 = HP0..HP3 , GP0/GP1 = HPC0/HPC1 , GP6 = LPD
set_property -dict [list \
  CONFIG.PSU__USE__M_AXI_GP0 {1} \
  CONFIG.PSU__MAXIGP0__DATA_WIDTH {32} \
  CONFIG.PSU__USE__M_AXI_GP1 {0} \
  CONFIG.PSU__USE__M_AXI_GP2 {0} \
  CONFIG.PSU__USE__S_AXI_GP0 {1} \
  CONFIG.PSU__SAXIGP0__DATA_WIDTH {128} \
  CONFIG.PSU__USE__S_AXI_GP1 {1} \
  CONFIG.PSU__SAXIGP1__DATA_WIDTH {128} \
  CONFIG.PSU__USE__S_AXI_GP2 {1} \
  CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
  CONFIG.PSU__USE__S_AXI_GP3 {1} \
  CONFIG.PSU__SAXIGP3__DATA_WIDTH {128} \
  CONFIG.PSU__USE__S_AXI_GP4 {1} \
  CONFIG.PSU__SAXIGP4__DATA_WIDTH {128} \
  CONFIG.PSU__USE__S_AXI_GP5 {1} \
  CONFIG.PSU__SAXIGP5__DATA_WIDTH {128} \
  CONFIG.PSU__USE__S_AXI_GP6 {1} \
  CONFIG.PSU__SAXIGP6__DATA_WIDTH {128} \
  CONFIG.PSU__FPGA_PL0_ENABLE {1} \
  CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100} \
  CONFIG.PSU__NUM_FABRIC_RESETS {1} \
] [get_bd_cells ps_e]

# ---------------- clocking ------------------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:6.0 clk_wiz_0
set_property -dict [list \
  CONFIG.RESET_TYPE {ACTIVE_LOW} \
  CONFIG.RESET_PORT {resetn} \
  CONFIG.CLKOUT1_USED {true} CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {100} \
  CONFIG.CLKOUT2_USED {true} CONFIG.CLKOUT2_REQUESTED_OUT_FREQ {300} \
  CONFIG.CLKOUT3_USED {true} CONFIG.CLKOUT3_REQUESTED_OUT_FREQ {600} \
  CONFIG.USE_LOCKED {true} \
] [get_bd_cells clk_wiz_0]
connect_bd_net [get_bd_pins ps_e/pl_clk0]    [get_bd_pins clk_wiz_0/clk_in1]
connect_bd_net [get_bd_pins ps_e/pl_resetn0] [get_bd_pins clk_wiz_0/resetn]

# one proc_sys_reset per published clock
for {set i 0} {$i < 3} {incr i} {
    create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_$i
    connect_bd_net [get_bd_pins clk_wiz_0/clk_out[expr {$i+1}]] \
                   [get_bd_pins proc_sys_reset_$i/slowest_sync_clk]
    connect_bd_net [get_bd_pins ps_e/pl_resetn0]  [get_bd_pins proc_sys_reset_$i/ext_reset_in]
    connect_bd_net [get_bd_pins clk_wiz_0/locked] [get_bd_pins proc_sys_reset_$i/dcm_locked]
}

# ---------------- interrupt controller (PFM.IRQ) ---------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_intc:4.1 axi_intc_0
set_property -dict [list CONFIG.C_IRQ_CONNECTION {1}] [get_bd_cells axi_intc_0]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out1]        [get_bd_pins axi_intc_0/s_axi_aclk]
connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] [get_bd_pins axi_intc_0/s_axi_aresetn]
connect_bd_net [get_bd_pins axi_intc_0/irq]            [get_bd_pins ps_e/pl_ps_irq0]

# ---------------- AXI-Lite control interconnect ----------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 interconnect_axilite
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] [get_bd_cells interconnect_axilite]
connect_bd_intf_net [get_bd_intf_pins ps_e/M_AXI_HPM0_FPD] \
                    [get_bd_intf_pins interconnect_axilite/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins interconnect_axilite/M00_AXI] \
                    [get_bd_intf_pins axi_intc_0/s_axi]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out1] [get_bd_pins interconnect_axilite/ACLK]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out1] [get_bd_pins interconnect_axilite/S00_ACLK]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out1] [get_bd_pins interconnect_axilite/M00_ACLK]
connect_bd_net [get_bd_pins proc_sys_reset_0/interconnect_aresetn] [get_bd_pins interconnect_axilite/ARESETN]
connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn]   [get_bd_pins interconnect_axilite/S00_ARESETN]
connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn]   [get_bd_pins interconnect_axilite/M00_ARESETN]
# M_AXI_HPM0_FPD must be clocked by the SAME source as the interconnect
# that consumes it, or validation fails with:
#   ERROR: [BD 41-237] Bus Interface property CLK_DOMAIN does not match
#          between .../auto_pc/S_AXI(clk_wiz_0_clk_out1)
#          and /ps_e/M_AXI_HPM0_FPD(ps_e_pl_clk0)
# so drive it from clk_out1, NOT pl_clk0.
connect_bd_net [get_bd_pins clk_wiz_0/clk_out1] [get_bd_pins ps_e/maxihpm0_fpd_aclk]

# HP/HPC/LPD slave clocks all run on the 300 MHz kernel clock.
# (The zynq_ultra_ps_e pin names are saxihp*_fpd_aclk / saxihpc*_fpd_aclk /
# saxi_lpd_aclk -- there are no 'saxigp*_aclk' pins.)
foreach p {saxihpc0_fpd_aclk saxihpc1_fpd_aclk saxihp0_fpd_aclk saxihp1_fpd_aclk saxihp2_fpd_aclk saxihp3_fpd_aclk saxi_lpd_aclk} {
    catch { connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins ps_e/$p] }
}

assign_bd_address
validate_bd_design -force
save_bd_design

# ---------------- PFM declarations (what makes it EXTENSIBLE) --------
set_property PFM_NAME "xilinx.com:xd:${PNAME}:${VER}" [get_files [current_bd_design].bd]
set_property PFM.IRQ {intr {id 0 range 32}} [get_bd_cells /axi_intc_0]
set_property PFM.CLOCK { \
  clk_out1 {id "0" is_default "true"  proc_sys_reset "proc_sys_reset_0" status "fixed"} \
  clk_out2 {id "1" is_default "false" proc_sys_reset "proc_sys_reset_1" status "fixed"} \
  clk_out3 {id "2" is_default "false" proc_sys_reset "proc_sys_reset_2" status "fixed"} \
} [get_bd_cells /clk_wiz_0]

set_property PFM.AXI_PORT { \
  S_AXI_HPC0_FPD {memport "S_AXI_HPC" sptag "HPC0" memory "ps_e HPC0_DDR_LOW"} \
  S_AXI_HPC1_FPD {memport "S_AXI_HPC" sptag "HPC1" memory "ps_e HPC1_DDR_LOW"} \
  S_AXI_HP0_FPD  {memport "S_AXI_HP"  sptag "HP0"  memory "ps_e HP0_DDR_LOW"} \
  S_AXI_HP1_FPD  {memport "S_AXI_HP"  sptag "HP1"  memory "ps_e HP1_DDR_LOW"} \
  S_AXI_HP2_FPD  {memport "S_AXI_HP"  sptag "HP2"  memory "ps_e HP2_DDR_LOW"} \
  S_AXI_HP3_FPD  {memport "S_AXI_HP"  sptag "HP3"  memory "ps_e HP3_DDR_LOW"} \
  S_AXI_LPD      {memport "S_AXI_HP"  sptag "LPD"  memory "ps_e LPD_DDR_LOW"} \
} [get_bd_cells /ps_e]

set parVal []
for {set i 1} {$i < 64} {incr i} {
    lappend parVal M[format %02d $i]_AXI {memport "M_AXI_GP"}
}
set_property PFM.AXI_PORT $parVal [get_bd_cells /interconnect_axilite]

set_property platform.default_output_type      "sd_card" [current_project]
set_property platform.design_intent.embedded   "true"    [current_project]
set_property platform.extensible               "true"    [current_project]
set_property platform.design_intent.server_managed "false" [current_project]
set_property platform.design_intent.external_host  "false" [current_project]
set_property platform.design_intent.datacenter     "false" [current_project]

validate_bd_design -force
save_bd_design

make_wrapper -files [get_files $PRJ/$PNAME.srcs/sources_1/bd/top/top.bd] -top -force
add_files -norecurse $PRJ/$PNAME.gen/sources_1/bd/top/hdl/top_wrapper.v
set_property top top_wrapper [current_fileset]
update_compile_order -fileset sources_1

# Generate BD targets EXPLICITLY and SINGLE-THREADED first.
#   ERROR: [Common 17-232] Could not create slave interpreter '::ipgen_iptclns'
#   CRITICAL WARNING: [IP_Flow 19-1747] Failed to deliver
#       '...zynq_ultra_ps_e_v3_4.ttcl'
# hit twice in a row on ps_e only (every other IP generated fine). That
# error is a Tcl sub-interpreter spawn failure, usually from the parallel
# IP-generation threads, so force serial generation here.
set_param general.maxThreads 1
generate_target all [get_files $PRJ/$PNAME.srcs/sources_1/bd/top/top.bd]

# ---------------- synthesise + export EXTENSIBLE XSA -----------------
launch_runs synth_1 -jobs 1
wait_on_run synth_1
puts "SYNTH: [get_property STATUS [get_runs synth_1]]"
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} { puts "SYNTH FAILED"; return }

open_run synth_1 -name synth_1
# -fixed is NOT used: this must be an expandable/extensible platform
write_hw_platform -force $OUT/${PNAME}.xsa
puts "XSA: $OUT/${PNAME}.xsa"
validate_hw_platform $OUT/${PNAME}.xsa
puts "PLATFORM BUILD DONE"
