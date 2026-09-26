# JTAG programming + ILA capture sequence. Verified working in PROJECT_HISTORY.md
# sections 17-19. Run INTERACTIVELY (-mode tcl), not batch -- you need to arm the
# trigger, then run the test on the board, then upload.
#
#   & "C:\Xilinx\Vivado\2022.2\bin\vivado.bat" -mode tcl
#
# --- Connect + program ---
open_hw_manager
connect_hw_server
open_hw_target
current_hw_device [get_hw_devices xck26_0]
set_property PROGRAM.FILE {C:/Xilinx/projects/kv260_custom_noDPU/KV260.runs/impl_1_01/top_wrapper.bit} [current_hw_device]
program_hw_devices [current_hw_device]

# --- Probes file (needs the design open too) ---
# open_project C:/Xilinx/projects/kv260_custom_noDPU/KV260.xpr
# open_run impl_1_01
# write_debug_probes -force C:/Xilinx/probes.ltx
# set_property PROBES.FILE {C:/Xilinx/probes.ltx} [current_hw_device]
# refresh_hw_device [current_hw_device]
# get_hw_probes -of_objects [get_hw_ilas]

# --- Arm trigger (adjust probe name to what get_hw_probes actually returns;
#     NOTE the stale-naming quirk: nets wired to hp2 may be NAMED hp0) ---
# set_property TRIGGER_COMPARE_VALUE eq1'b1 [get_hw_probes top_i/smartconnect_hp2/M00_AXI_awvalid]
# run_hw_ila [get_hw_ilas]
#
# --- NOW run the test on the board, THEN: ---
# wait_on_hw_ila [get_hw_ilas]
# upload_hw_ila_data [get_hw_ilas]
# set ila_data [current_hw_ila_data]
# write_hw_ila_data -force -csv_file C:/Xilinx/ila_capture.csv $ila_data
#
# Check STATUS.CORE_STATUS via `report_property [get_hw_ilas]`:
#   "FULL"                = trigger fired, data captured
#   "WAITING FOR TRIGGER" = never fired -- you're watching the wrong wire
