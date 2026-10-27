# v++ post-system-link hook (build_overlay.sh): turn off the register slices and data FIFOs v++ puts on
# every port of the AXI interconnects in front of the PS's HP ports. With them, our 32-bit kernel ports are
# upsized in FIFO mode and the interconnects cost ~25 BRAM tiles, which the KV260 overlay cannot spare.
# Same idea as the DPU TRD's syslink/strip_interconnects.tcl, which looks for its own platform's
# interconnect names and finds none in ours.
foreach ic [get_bd_cells -hierarchical -quiet -filter {VLNV =~ "xilinx.com:ip:axi_interconnect:*"}] {
  # v++ builds them with STRATEGY 2 (maximize performance), which locks the slave-side data FIFOs on
  # ("disabled parameter ... ignored"); the custom strategy (0) unlocks them.
  set_property CONFIG.STRATEGY 0 $ic
  set props {}
  for {set i 0} {$i < [get_property CONFIG.NUM_SI $ic]} {incr i} {
    lappend props CONFIG.S[format %02d $i]_HAS_REGSLICE 0 CONFIG.S[format %02d $i]_HAS_DATA_FIFO 0
  }
  for {set i 0} {$i < [get_property CONFIG.NUM_MI $ic]} {incr i} {
    lappend props CONFIG.M[format %02d $i]_HAS_REGSLICE 0 CONFIG.M[format %02d $i]_HAS_DATA_FIFO 0
  }
  set_property -dict $props $ic
  puts "strip_interconnects: $ic"
}
