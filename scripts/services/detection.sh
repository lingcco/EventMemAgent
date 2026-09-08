#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/common.sh"

model_path="$(resolve_repo_path "${DETECTION_MODEL_PATH:-checkpoints/tools/grounding-dino-base}")"
require_path "$model_path" "detection model"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DETECTION_CUDA_VISIBLE_DEVICES:-0}}"

exec python -m eventmemagent.services.detection_server \
  --model-path "$model_path" \
  --device "${DEVICE:-cuda}" --host "${HOST:-0.0.0.0}" --port "${PORT:-8090}" \
  "$@"
