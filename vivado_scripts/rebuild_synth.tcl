# Reconstructed from the commands that were verified working in PROJECT_HISTORY.md.
# Run from Windows:
#   & "C:\Xilinx\Vivado\2022.2\bin\vivado.bat" -mode batch -source rebuild_synth.tcl -log synth.log
# NOTE: free RAM first (peaks ~10.5GB). Close browsers/other apps. Let it run
# UNINTERRUPTED -- interrupting mid-run loses all progress (learned the hard way).
open_project C:/Xilinx/projects/kv260_custom_noDPU/KV260.xpr
reset_run synth_1
launch_runs synth_1 -jobs 1
wait_on_run synth_1
puts "SYNTH STATUS: [get_property STATUS [get_runs synth_1]]"
