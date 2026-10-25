#!/usr/bin/env bash
# Run a command inside the Vitis AI image with the repo mounted at /workspace.
#   deploy/vitis_ai/run_docker.sh python quantize.py --model pwcnet --mode inspect --videos ...
# Build the image first: docker build -t kria-vitis-ai:3.5 deploy/vitis_ai
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
IMAGE=${IMAGE:-kria-vitis-ai:3.5}
MOUNTS=(-v "$REPO:/workspace")
[[ -f "$HOME/.netrc" ]] && MOUNTS+=(-v "$HOME/.netrc:/run/netrc:ro")  # W&B credentials
mkdir -p "$HOME/.cache/torch" && MOUNTS+=(-v "$HOME/.cache/torch:/cache/torch")  # reuse downloaded checkpoints

# Mount points outside $HOME: Docker would create them root-owned and make $HOME unwritable.
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp/home -e TORCH_HOME=/cache/torch -e MPLCONFIGDIR=/tmp/home/mpl \
  -e COLLISION_WANDB -e COLLISION_WANDB_PROJECT -e ARCH -e IN_DIR -e OUT_DIR "${MOUNTS[@]}" -w /workspace/deploy/vitis_ai "$IMAGE" \
  bash -c 'mkdir -p "$HOME" && { [ ! -f /run/netrc ] || cp /run/netrc "$HOME/.netrc"; } \
    && source /opt/vitis_ai/conda/etc/profile.d/conda.sh && conda activate vitis-ai-pytorch && "$@"' _ "$@"
