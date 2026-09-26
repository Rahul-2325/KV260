# Reconstructed from the commands that were verified working in PROJECT_HISTORY.md.
# Run AFTER synthesis completes:
#   & "C:\Xilinx\Vivado\2022.2\bin\vivado.bat" -mode batch -source rebuild_impl.tcl -log impl.log
# CRITICAL: the "-to_step write_bitstream" flag is REQUIRED. Omitting it causes the
# run to stop at phys_opt_design without ever producing a bitstream (this actually
# happened once and cost a full rebuild cycle to notice).
open_project C:/Xilinx/projects/kv260_custom_noDPU/KV260.xpr
launch_runs impl_1_01 -to_step write_bitstream -jobs 1
wait_on_run impl_1_01
puts "IMPL STATUS: [get_property STATUS [get_runs impl_1_01]]"
