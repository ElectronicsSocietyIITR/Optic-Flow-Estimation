#!/usr/bin/env bash
# Build the FCW overlay for the KV260 (Linux; Vitis / Vivado 2023.1, the version AMD's Kria platforms, the
# Vitis AI 3.5 DPU and the Kria Ubuntu 22.04 XRT line up on). Stages are skipped when their output exists:
#   1 hardware platform (kv260_platform.tcl)  -> kv260_fcw.xsa
#   2 Vitis platform (xsct)                   -> pfm/kv260_fcw/export/kv260_fcw/kv260_fcw.xpfm
#   3 DPU B3136 kernel (AMD DPU TRD)          -> dpu.xo  (+ arch.json for the YOLO compile)
#   4 our kernels (v++ -c)                    -> preprocess.xo, lk_wrapper.xo, ttc_engine_top.xo
#   5 link (v++ -l, link.cfg)                 -> fcw.xclbin
#   6 Kria app (xmutil loadapp fcw)           -> app/fcw/{fcw.bit.bin, fcw.dtbo, fcw.xclbin, shell.json}
# Usage: XILINX=/tools/Xilinx TRD=/opt/fcw/DPUCZDX8G_VAI_v3.0 OUT=/opt/fcw/build ./build_overlay.sh
set -euo pipefail
XILINX=${XILINX:-/tools/Xilinx}
VER=${VER:-2023.1}
TRD=${TRD:-/opt/fcw/DPUCZDX8G_VAI_v3.0}
OUT=${OUT:-/opt/fcw/build}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
VISION=${VISION:-$REPO/third_party/Vitis_Libraries/vision/L1/include}
JOBS=${JOBS:-4}  # Vivado jobs: keep low on a 16 GB machine
source "$XILINX/Vitis/$VER/settings64.sh"
mkdir -p "$OUT" && cd "$OUT"

# 1. Hardware platform
[[ -f hw/kv260_fcw.xsa ]] || vivado -mode batch -notrace -source "$REPO/hls/system/kv260_platform.tcl" -tclargs "$OUT/hw"
need() { [[ -f $1 ]] || { echo "stage failed: $1 was not produced" >&2; exit 1; }; }  # Vivado / xsct exit 0 on script errors
need hw/kv260_fcw.xsa

# 2. Vitis platform (no boot components: Kria boots from its own image and loads the overlay with xmutil)
XPFM=$OUT/pfm/kv260_fcw/export/kv260_fcw/kv260_fcw.xpfm
if [[ ! -f $XPFM ]]; then
  mkdir -p pfm/boot pfm/sd_dir
  cat > pfm/make_pfm.tcl <<EOF
setws $OUT/pfm
platform create -name kv260_fcw -hw $OUT/hw/kv260_fcw.xsa -out $OUT/pfm -no-boot-bsp
domain create -name xrt -proc psu_cortexa53 -os linux -arch {64-bit} -runtime {ocl} -bootmode {sd}
domain config -boot $OUT/pfm/boot
domain config -image $OUT/pfm/sd_dir
platform generate
EOF
  xsct pfm/make_pfm.tcl
fi
need "$XPFM"

# 3. DPU B3136 with UltraRAM (TRD README: B3136 U_BANK_IMG 4, WGT 15, BIAS 1), so BRAM stays free for our kernels
if [[ ! -f dpu/dpu.xo ]]; then
  rm -rf dpu && cp -r "$TRD/prj/Vitis" dpu && cp -r "$TRD/dpu_ip" dpu_ip
  sed -i -e 's/^`define B4096 *$/`define B3136/' -e 's/^`define URAM_DISABLE *$/`define URAM_ENABLE/' \
         -e '/`ifdef URAM_ENABLE/,/`elsif URAM_DISABLE/{s/def_UBANK_IMG_N *[0-9]*/def_UBANK_IMG_N          4/;s/def_UBANK_WGT_N *[0-9]*/def_UBANK_WGT_N          15/}' \
         dpu/dpu_conf.vh
  grep -E '^`define (B[0-9]+|URAM_)' dpu/dpu_conf.vh
  (cd dpu && make binary_container_1/dpu.xo KERNEL=DPU DEVICE=kv260 TARGET=hw VIVADO_ROOT="$XILINX/Vivado/$VER" \
     DIR_PRJ="$OUT/dpu" TRD_HOME="$OUT")
  cp dpu/binary_container_1/dpu.xo dpu/dpu.xo
fi
need dpu/dpu.xo

# 4. Our HLS kernels (sources and options as in each kernel's hls_config.cfg)
hlsxo() {  # <name> <top> <source> [extra flags...]
  local name=$1 top=$2 src=$3; shift 3
  [[ -f $name.xo ]] || v++ -c -t hw --platform "$XPFM" -k "$top" --hls.clock 250000000:"$top" \
    -I "$REPO/hls" -I "$VISION" "$@" "$src" -o "$name.xo" --temp_dir "_x_$name" --save-temps
}
hlsxo preprocess preprocess "$REPO/hls/preprocess/src/preprocess.cpp"
hlsxo lk_wrapper lk_wrapper "$REPO/hls/lk_wrapper/src/lk_wrapper.cpp" -DLK_MAX=512 -DLK_LINES=34 -DLK_WIN=11 -DLK_MEDIAN=3
hlsxo ttc_engine_top ttc_engine_top "$REPO/hls/ttc_engine/src/ttc_engine.cpp" -DTTC_LANES=4
for x in preprocess lk_wrapper ttc_engine_top; do need $x.xo; done

# 5. Link
[[ -f fcw.xclbin ]] || v++ -l -t hw --platform "$XPFM" --config "$REPO/hls/system/link.cfg" \
  --vivado.impl.jobs "$JOBS" --vivado.synth.jobs "$JOBS" --save-temps --temp_dir _x_link \
  dpu/dpu.xo preprocess.xo lk_wrapper.xo ttc_engine_top.xo -o fcw.xclbin
need fcw.xclbin
cp _x_link/link/vivado/vpl/prj/prj.gen/sources_1/bd/*/ip/*_DPUCZDX8G_1_0/arch.json arch.json 2>/dev/null || true

# 6. Kria app: bitstream as .bin, device-tree overlay with the zocl node, shell.json
if [[ ! -f app/fcw/fcw.xclbin ]]; then
  mkdir -p app/fcw dt
  bit=$(ls _x_link/link/int/*.bit | head -1)
  echo "all: { $bit }" > app/fcw.bif && bootgen -image app/fcw.bif -arch zynqmp -process_bitstream bin -o app/fcw/fcw.bit.bin -w on
  [[ -f app/fcw/fcw.bit.bin ]] || cp "$bit.bin" app/fcw/fcw.bit.bin
  xsct -eval "createdts -hw $OUT/hw/kv260_fcw.xsa -zocl -out $OUT/dt -platform-name fcw -git-branch xlnx_rel_v$VER -overlay -compile"
  dtc -@ -O dtb -o app/fcw/fcw.dtbo dt/fcw/psu_cortexa53_0/device_tree_domain/bsp/pl.dtsi
  cp fcw.xclbin app/fcw/fcw.xclbin
  printf '{\n  "shell_type" : "XRT_FLAT",\n  "num_slots": "1"\n}\n' > app/fcw/shell.json
fi
echo "overlay: $OUT/app/fcw  (copy to /lib/firmware/xilinx/fcw on the board; sudo xmutil loadapp fcw)"
echo "DPU arch for the YOLO compile: $OUT/arch.json"
