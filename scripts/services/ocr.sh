#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/common.sh"

model_path="$(resolve_repo_path "${OCR_MODEL_PATH:-checkpoints/tools/DeepSeek-OCR-updated}")"
require_path "$model_path" "OCR model"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${OCR_CUDA_VISIBLE_DEVICES:-0}}"

exec python -m eventmemagent.services.ocr_server \
  --model-path "$model_path" \
  --dtype "${DTYPE:-bfloat16}" --device "${DEVICE:-cuda}" \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-8089}" \
  "$@"
