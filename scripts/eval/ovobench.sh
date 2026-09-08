#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/common.sh"
configure_eval_devices

MODEL_PATH="$(resolve_repo_path "${MODEL_PATH:-checkpoints/released/EventMemAgent-8B}")"
EMBEDDING_MODEL_PATH="$(resolve_repo_path "${EMBEDDING_MODEL_PATH:-checkpoints/pretrained/Qwen3-Embedding-0.6B}")"
benchmark="${EVENTMEMAGENT_BENCHMARK:-ovobench}"
if [[ "$benchmark" == "streamingbench" ]]; then
  OVOBENCH_DATA="$(resolve_repo_path "${STREAMINGBENCH_PARQUET:-data/eval/streamingbench/eventmemagent_streamingbench.parquet}")"
elif [[ "$benchmark" == "ovobench" ]]; then
  OVOBENCH_DATA="$(resolve_repo_path "${OVOBENCH_DATA:-data/eval/ovobench/eventmemagent_ovobench.parquet}")"
else
  echo "Unknown parquet benchmark: $benchmark" >&2
  exit 2
fi
TOOL_CONFIG_PATH="$(resolve_repo_path "${TOOL_CONFIG_PATH:-configs/tools.yaml}")"
EVAL_OUTPUT_ROOT="$(resolve_repo_path "${EVAL_OUTPUT_ROOT:-runs/eval}")"
OUTPUT_DIR="${OUTPUT_DIR:-${EVAL_OUTPUT_ROOT}/${benchmark}/$(date +%Y%m%d_%H%M%S)}"
MODE="${MODE:-batch}"

require_path "$MODEL_PATH" "trained model"
require_path "$EMBEDDING_MODEL_PATH" "embedding model"
require_file "$OVOBENCH_DATA" "$benchmark parquet"
require_file "$TOOL_CONFIG_PATH" "tool config"
mkdir -p "$OUTPUT_DIR"

if [[ "$MODE" == "batch" ]]; then
  module="eventmemagent.evaluation.batch"
elif [[ "$MODE" == "normal" ]]; then
  module="eventmemagent.evaluation.run"
else
  echo "MODE must be 'batch' or 'normal', got: $MODE" >&2
  exit 2
fi

args=(
  --parquet_path "$OVOBENCH_DATA"
  --output_dir "$OUTPUT_DIR"
  --model_path "$MODEL_PATH"
  --tool_config_path "$TOOL_CONFIG_PATH"
  --embedding_model_path "$EMBEDDING_MODEL_PATH"
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE:-1}"
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION:-0.7}"
  --max_model_len "${MAX_MODEL_LEN:-32768}"
  --max_turns "${MAX_TURNS:-10}"
  --max_new_tokens "${MAX_NEW_TOKENS:-16384}"
  --temperature "${TEMPERATURE:-0}"
  --save_interval "${SAVE_INTERVAL:-30}"
  --start_idx "${START_IDX:-0}"
)

if [[ -n "${END_IDX:-}" ]]; then
  args+=(--end_idx "$END_IDX")
fi
if [[ "$MODE" == "batch" ]]; then
  args+=(--batch_size "${BATCH_SIZE:-8}")
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${EVAL_CUDA_VISIBLE_DEVICES:-0}}"
python -m "$module" "${args[@]}" "$@"
