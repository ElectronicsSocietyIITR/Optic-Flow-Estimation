#!/usr/bin/env bash
# Compile the quantized xmodels for the KV260 DPU. Run inside the Vitis AI 3.5 docker
# after `quantize.py --mode test` has produced quantized/<model>/*_int.xmodel.
#   ./compile.sh                 # both models
#   ./compile.sh yolov9t         # selected models
set -euo pipefail

# Default: B3136, the DPU size that fits next to our kernels on the K26 (docs/superpowers/reports/2026-10-fabric-budget.md).
# For the board, pass the arch.json the DPU build writes (ARCH=/path/to/arch.json): its fingerprint is the exact DPU.
ARCH=${ARCH:-$(dirname "$0")/arch_b3136.json}
IN_DIR=${IN_DIR:-quantized}
OUT_DIR=${OUT_DIR:-compiled}

MODELS=("$@"); [[ ${#MODELS[@]} -gt 0 ]] || MODELS=(pwcnet yolov9t)
for model in "${MODELS[@]}"; do
  xmodel=$(ls "$IN_DIR/$model"/*_int.xmodel)
  vai_c_xir --xmodel "$xmodel" --arch "$ARCH" --output_dir "$OUT_DIR/$model" --net_name "${model}_kv260"
  # vai_c_xir exits 0 even when the compiler aborts, so check for its output.
  [[ -f "$OUT_DIR/$model/${model}_kv260.xmodel" ]] || { echo "compilation of $model failed" >&2; exit 1; }
  # How the graph was partitioned: every CPU subgraph is a DPU<->CPU round trip at runtime.
  xir subgraph "$OUT_DIR/$model/${model}_kv260.xmodel" | tee "$OUT_DIR/$model/subgraphs.txt"
done
