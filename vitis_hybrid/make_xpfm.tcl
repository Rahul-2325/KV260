# make_xpfm.tcl -- run with xsct
#
# Turns the extensible XSA (out/kv260_ext.xsa) into a Vitis platform
# (.xpfm) that v++ --link will accept.
#
# We only need the platform for LINKING (v++ -l dpu.xo + lcam.xo -> .xclbin).
# We do NOT need --package, because the board already boots a working
# PetaLinux image and loads PL designs through xmutil/dtbo. So:
#   -no-boot-bsp  : skip the FSBL/PMUFW build (needs no BSP sources)
#   sd_card image / rootfs / sysroot are deliberately NOT set -- those are
#   only consumed by v++ --package, which we bypass.
#
# The domain MUST be linux + ocl runtime, otherwise v++ reports the
# platform as non-accelerated (the exact failure we had with
# kv260_9ip_v2.xpfm: platforminfo listed no clocks and no AXI ports).

set root C:/Xilinx/projects/kv260_ext_pfm
setws $root/vitis_ws

platform create -name kv260_ext_pfm \
                -hw $root/out/kv260_ext.xsa \
                -no-boot-bsp

domain create -name xrt -os linux -proc psu_cortexa53 -runtime {ocl}
domain active xrt

platform write
platform generate

puts "XPFM: $root/vitis_ws/kv260_ext_pfm/export/kv260_ext_pfm/kv260_ext_pfm.xpfm"
puts "MAKE_XPFM DONE"
