#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/common.sh"

# The archived benchmark experiments use precomputed 4B memories.
# Raw-video evaluation is retained explicitly as INPUT_MODE=online.
if [[ "${INPUT_MODE:-parquet}" == "parquet" ]]; then
  export EVENTMEMAGENT_BENCHMARK=streamingbench
  exec bash "${EVENTMEMAGENT_ROOT}/scripts/eval/ovobench.sh" "$@"
elif [[ "$INPUT_MODE" != "online" ]]; then
  echo "INPUT_MODE must be parquet or online." >&2
  exit 2
fi
configure_eval_devices

MODEL_PATH="$(resolve_repo_path "${MODEL_PATH:-checkpoints/released/EventMemAgent-8B}")"
EMBEDDING_MODEL_PATH="$(resolve_repo_path "${EMBEDDING_MODEL_PATH:-checkpoints/pretrained/Qwen3-Embedding-0.6B}")"
STREAMINGBENCH_DATA="$(resolve_repo_path "${STREAMINGBENCH_DATA:-data/eval/streamingbench/questions_real.json}")"
TOOL_CONFIG_PATH="$(resolve_repo_path "${TOOL_CONFIG_PATH:-configs/tools.yaml}")"
EVAL_OUTPUT_ROOT="$(resolve_repo_path "${EVAL_OUTPUT_ROOT:-runs/eval}")"
OUTPUT_FILE="${OUTPUT_FILE:-${EVAL_OUTPUT_ROOT}/streamingbench/eventmemagent_$(date +%Y%m%d_%H%M%S).json}"

require_path "$MODEL_PATH" "trained model"
require_path "$EMBEDDING_MODEL_PATH" "embedding model"
require_file "$STREAMINGBENCH_DATA" "StreamingBench questions"
require_file "$TOOL_CONFIG_PATH" "tool config"
mkdir -p "$(dirname "$OUTPUT_FILE")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${EVAL_CUDA_VISIBLE_DEVICES:-0}}"

python -m eventmemagent.evaluation.streamingbench \
  --data_file "$STREAMINGBENCH_DATA" \
  --output_file "$OUTPUT_FILE" \
  --context_time "${CONTEXT_TIME:--1}" \
  --model_name "${MODEL_NAME:-EventMemAgent}" \
  --model_path "$MODEL_PATH" \
  --tool_config_path "$TOOL_CONFIG_PATH" \
  --embedding_model_path "$EMBEDDING_MODEL_PATH" \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --num_workers "${NUM_WORKERS:-1}" \
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE:-1}" \
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION:-0.7}" \
  --max_model_len "${MAX_MODEL_LEN:-32768}" \
  --max_turns "${MAX_TURNS:-5}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-16384}" \
  --temperature "${TEMPERATURE:-0}" \
  "$@"
