# Minimal KV260 extensible hardware platform for the FCW overlay (Vivado 2023.1, batch):
#   vivado -mode batch -source kv260_platform.tcl -tclargs <out_dir>
# Follows AMD's Vitis-Tutorials 2023.1 Vitis_Platform_Creation/Design_Tutorials/01-Edge-KV260 (step 1), with
# our clocks: 100 MHz control, 250 MHz kernels (default; preprocess / LK route at 261 / 279 MHz), 300 and
# 600 MHz for the DPU (aclk, ap_clk_2). No video pipeline: capture is USB or file, display the PS DisplayPort.
set out [lindex $argv 0]
if {$out eq ""} { set out [pwd]/kv260_fcw_hw }
file mkdir $out

# KV260 board files: Vivado 2023.1 ships them (data/xhub/boards/XilinxBoardStore/boards/Xilinx/kv260_*); the
# online board store no longer has a 2023.1 branch, so no catalog refresh.
set_param board.repoPaths [file join $::env(XILINX_VIVADO) data xhub boards XilinxBoardStore boards]

set som     [lindex [get_board_parts -latest_file_version *kv260_som*] 0]
set carrier [lindex [get_board_components -quiet *kv260_carrier*som240_1_connector*] 0]
puts "board part: $som  carrier: $carrier"

create_project -force kv260_fcw $out/vivado -part xck26-sfvc784-2LV-c
set_property board_part $som [current_project]
if {$carrier ne ""} { set_property board_connections "som240_1_connector $carrier" [current_project] }
set_property platform.extensible true [current_project]
create_bd_design kv260_fcw

# Processor system with the KV260 board preset; HPM0_LPD controls the interrupt controller, HPM0/1_FPD
# are left to the kernels (exported below).
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e zynq_ultra_ps_e_0]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $ps
set_property -dict [list CONFIG.PSU__USE__M_AXI_GP0 {0} CONFIG.PSU__USE__M_AXI_GP1 {0} CONFIG.PSU__USE__M_AXI_GP2 {1} \
  CONFIG.PSU__USE__IRQ0 {1}] $ps

# Clocks and resets.
set ck [create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz clk_wiz_0]
set_property -dict [list CONFIG.CLKOUT2_USED {true} CONFIG.CLKOUT3_USED {true} CONFIG.CLKOUT4_USED {true} \
  CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {100} CONFIG.CLKOUT2_REQUESTED_OUT_FREQ {250} \
  CONFIG.CLKOUT3_REQUESTED_OUT_FREQ {300} CONFIG.CLKOUT4_REQUESTED_OUT_FREQ {600} \
  CONFIG.RESET_TYPE {ACTIVE_LOW} CONFIG.RESET_PORT {resetn}] $ck
connect_bd_net [get_bd_pins $ps/pl_clk0] [get_bd_pins $ck/clk_in1]
connect_bd_net [get_bd_pins $ps/pl_resetn0] [get_bd_pins $ck/resetn]
foreach i {1 2 3 4} {
  set r [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset proc_sys_reset_$i]
  connect_bd_net [get_bd_pins $ck/clk_out$i] [get_bd_pins $r/slowest_sync_clk]
  connect_bd_net [get_bd_pins $ps/pl_resetn0] [get_bd_pins $r/ext_reset_in]
  connect_bd_net [get_bd_pins $ck/locked] [get_bd_pins $r/dcm_locked]
}

# Interrupt controller for the kernels (v++ connects kernel interrupts to it).
set intc [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_intc axi_intc_0]
set_property -dict [list CONFIG.C_IRQ_CONNECTION {1} CONFIG.C_ASYNC_INTR {0xFFFFFFFF}] $intc
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list Clk_master {/clk_wiz_0/clk_out1 (100 MHz)} \
  Clk_slave {/clk_wiz_0/clk_out1 (100 MHz)} Clk_xbar {/clk_wiz_0/clk_out1 (100 MHz)} \
  Master {/zynq_ultra_ps_e_0/M_AXI_HPM0_LPD} Slave {/axi_intc_0/s_axi} intc_ip {New AXI Interconnect} master_apm {0}] \
  [get_bd_intf_pins $intc/s_axi]
connect_bd_net [get_bd_pins $intc/irq] [get_bd_pins $ps/pl_ps_irq0]

# Platform interfaces (PFM properties): clocks, AXI ports, interrupts.
set_property PFM_NAME {xilinx.com:fcw:kv260_fcw:1.0} [get_files [current_bd_design].bd]
set_property PFM.CLOCK {clk_out1 {id "0" is_default "false" proc_sys_reset "/proc_sys_reset_1" status "fixed"} \
  clk_out2 {id "1" is_default "true" proc_sys_reset "/proc_sys_reset_2" status "fixed"} \
  clk_out3 {id "2" is_default "false" proc_sys_reset "/proc_sys_reset_3" status "fixed"} \
  clk_out4 {id "3" is_default "false" proc_sys_reset "/proc_sys_reset_4" status "fixed"}} $ck
set_property PFM.AXI_PORT {M_AXI_HPM0_FPD {memport "M_AXI_GP" sptag "" memory ""} \
  M_AXI_HPM1_FPD {memport "M_AXI_GP" sptag "" memory ""} \
  S_AXI_HPC0_FPD {memport "S_AXI_HP" sptag "HPC0" memory ""} S_AXI_HPC1_FPD {memport "S_AXI_HP" sptag "HPC1" memory ""} \
  S_AXI_HP0_FPD {memport "S_AXI_HP" sptag "HP0" memory ""} S_AXI_HP1_FPD {memport "S_AXI_HP" sptag "HP1" memory ""} \
  S_AXI_HP2_FPD {memport "S_AXI_HP" sptag "HP2" memory ""} S_AXI_HP3_FPD {memport "S_AXI_HP" sptag "HP3" memory ""}} $ps
set_property PFM.IRQ {intr {id 0 range 32}} $intc

assign_bd_address
validate_bd_design
save_bd_design
make_wrapper -files [get_files [current_bd_design].bd] -top -import
generate_target all [get_files [current_bd_design].bd]

set_property platform.default_output_type "sd_card" [current_project]
set_property platform.design_intent.embedded "true" [current_project]
set_property platform.design_intent.server_managed "false" [current_project]
set_property platform.design_intent.external_host "false" [current_project]
set_property platform.design_intent.datacenter "false" [current_project]
set_property platform.name kv260_fcw [current_project]

launch_runs synth_1 -jobs 8
wait_on_run synth_1
write_hw_platform -force -file $out/kv260_fcw.xsa
puts "XSA: $out/kv260_fcw.xsa"
