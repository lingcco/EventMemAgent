#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/common.sh"
configure_eval_devices

MODEL_PATH="$(resolve_repo_path "${MODEL_PATH:-checkpoints/released/EventMemAgent-8B}")"
EMBEDDING_MODEL_PATH="$(resolve_repo_path "${EMBEDDING_MODEL_PATH:-checkpoints/pretrained/Qwen3-Embedding-0.6B}")"
PARQUET_PATH="$(resolve_repo_path "${PARQUET_PATH:-data/eval/ovobench/eventmemagent_ovobench.parquet}")"
TOOL_CONFIG_PATH="$(resolve_repo_path "${TOOL_CONFIG_PATH:-configs/tools.yaml}")"

require_path "$MODEL_PATH" "model"
require_path "$EMBEDDING_MODEL_PATH" "embedding model"
require_file "$PARQUET_PATH" "offline inference parquet"
require_file "$TOOL_CONFIG_PATH" "tool config"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${EVAL_CUDA_VISIBLE_DEVICES:-0}}"

args=(
  --mode offline
  --model_path "$MODEL_PATH"
  --embedding_model_path "$EMBEDDING_MODEL_PATH"
  --tool_config_path "$TOOL_CONFIG_PATH"
  --parquet_path "$PARQUET_PATH"
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE:-1}"
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION:-0.7}"
  --max_model_len "${MAX_MODEL_LEN:-32768}"
  --max_turns "${MAX_TURNS:-5}"
  --max_new_tokens "${MAX_NEW_TOKENS:-2048}"
  --temperature "${TEMPERATURE:-0}"
)

if [[ -n "${1:-}" ]]; then
  args+=(--sample_idx "$1")
  shift
fi

python -m eventmemagent.inference.run "${args[@]}" "$@"
