#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/common.sh"
model_path="$(resolve_repo_path "${MEMORY_CAPTION_MODEL_PATH:-checkpoints/pretrained/Qwen3-VL-4B-Instruct}")"
require_file "$model_path/config.json" "4B caption model config"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${MEMORY_CUDA_VISIBLE_DEVICES:-0}}"
exec python -m vllm.entrypoints.openai.api_server \
  --model "$model_path" \
  --served-model-name "${MEMORY_CAPTION_MODEL_NAME:-Qwen3-VL-4B-Instruct}" \
  --host "${MEMORY_HOST:-127.0.0.1}" --port "${MEMORY_PORT:-8000}" \
  --tensor-parallel-size "${MEMORY_TENSOR_PARALLEL_SIZE:-1}" \
  --max-model-len 32768 --gpu-memory-utilization "${MEMORY_GPU_MEMORY_UTILIZATION:-0.8}" \
  --max-num-seqs 16 --mm-processor-cache-gb 0 \
  --limit-mm-per-prompt '{"image": 64, "video": 1}' "$@"
