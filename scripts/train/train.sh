#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/common.sh"

export BASE_MODEL_PATH="$(resolve_repo_path "${BASE_MODEL_PATH:-checkpoints/pretrained/Qwen3-VL-8B-Instruct}")"
export EMBEDDING_MODEL_PATH="$(resolve_repo_path "${EMBEDDING_MODEL_PATH:-checkpoints/pretrained/Qwen3-Embedding-0.6B}")"
export TRAIN_DATA="$(resolve_repo_path "${TRAIN_DATA:-data/train/eventmemagent_train.parquet}")"
export VAL_DATA="${VAL_DATA:-}"
if [[ -n "$VAL_DATA" ]]; then
  export VAL_DATA="$(resolve_repo_path "$VAL_DATA")"
fi
export TOOL_CONFIG_PATH="$(resolve_repo_path "${TOOL_CONFIG_PATH:-configs/tools.yaml}")"
export TRAIN_OUTPUT_DIR="$(resolve_repo_path "${TRAIN_OUTPUT_DIR:-runs/train/eventmemagent_qwen3vl_8b}")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
export NUM_GPUS="${NUM_GPUS:-8}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${TRAIN_TENSOR_PARALLEL_SIZE:-8}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

RESUME_MODE="${RESUME_MODE:-auto}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
SKIP_TOOL_SERVER_CHECK="${SKIP_TOOL_SERVER_CHECK:-0}"
no_validation=0
dry_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-validation) no_validation=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    *) break ;;
  esac
done

hydra_overrides=(
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_PARALLEL_SIZE}"
  "trainer.n_gpus_per_node=${NUM_GPUS}"
  "trainer.resume_mode=${RESUME_MODE}"
)
if [[ -n "$RESUME_FROM_PATH" ]]; then
  hydra_overrides+=("trainer.resume_from_path=$(resolve_repo_path "$RESUME_FROM_PATH")")
fi
if [[ "$no_validation" == 1 ]]; then
  # Explicitly ignore an optional VAL_DATA configured in .env.
  for override in "$@"; do
    case "${override#+}" in
      trainer.test_freq=*|trainer.val_before_train=*|trainer.val_only=*|data.val_files=*|data.val_max_samples=*)
        echo "Do not combine --no-validation with validation overrides: ${override}" >&2
        exit 2 ;;
    esac
  done
  export VAL_DATA=""
  hydra_overrides+=(
    "trainer.val_before_train=false"
    "trainer.test_freq=-1"
    "trainer.val_only=false"
    "data.val_files=[]"
  )
fi
if [[ "$dry_run" == 1 ]]; then
  exec python scripts/utils/print_train_config.py "${hydra_overrides[@]}" "$@"
fi

require_positive_integer "$NUM_GPUS" "NUM_GPUS"
require_positive_integer "$TENSOR_PARALLEL_SIZE" "TENSOR_PARALLEL_SIZE"

visible_gpu_count="$(count_csv_items "$CUDA_VISIBLE_DEVICES")"
if (( NUM_GPUS > visible_gpu_count )); then
  echo "NUM_GPUS=${NUM_GPUS}, but CUDA_VISIBLE_DEVICES exposes only ${visible_gpu_count} device(s): ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi
if (( TENSOR_PARALLEL_SIZE > NUM_GPUS )); then
  echo "TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE} cannot exceed NUM_GPUS=${NUM_GPUS}." >&2
  exit 2
fi
if (( NUM_GPUS % TENSOR_PARALLEL_SIZE != 0 )); then
  echo "NUM_GPUS=${NUM_GPUS} must be divisible by TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}." >&2
  exit 2
fi

case "$RESUME_MODE" in
  auto|disable) ;;
  resume_path)
    if [[ -z "$RESUME_FROM_PATH" ]]; then
      echo "RESUME_FROM_PATH is required when RESUME_MODE=resume_path." >&2
      exit 2
    fi
    RESUME_FROM_PATH="$(resolve_repo_path "$RESUME_FROM_PATH")"
    require_path "$RESUME_FROM_PATH" "resume checkpoint"
    ;;
  *)
    echo "RESUME_MODE must be auto, disable, or resume_path; got: ${RESUME_MODE}" >&2
    exit 2
    ;;
esac

require_path "$BASE_MODEL_PATH" "base model"
require_path "$EMBEDDING_MODEL_PATH" "embedding model"
require_file "$TRAIN_DATA" "training data"
if [[ -n "$VAL_DATA" ]]; then
  require_file "$VAL_DATA" "validation data"
fi
require_file "$TOOL_CONFIG_PATH" "tool config"
mkdir -p "$TRAIN_OUTPUT_DIR"

python scripts/utils/validate_setup.py --mode train
python scripts/utils/validate_setup.py --mode runtime

if [[ "$SKIP_TOOL_SERVER_CHECK" != "1" ]]; then
  require_http_health "${OCR_HEALTH_URL:-http://localhost:8089/health}" "OCR service"
  require_http_health "${DETECTION_HEALTH_URL:-http://localhost:8090/health}" "detection service"
fi

open_files_hard="$(ulimit -Hn)"
open_files_target=65535
if [[ "$open_files_hard" != unlimited ]] && (( open_files_hard < open_files_target )); then
  open_files_target="$open_files_hard"
  echo "Warning: server hard limit allows only ${open_files_target} open files (recommended: 65535)." >&2
fi
ulimit -Sn "$open_files_target"
export RAY_DEBUG_POST_MORTEM="${RAY_DEBUG_POST_MORTEM:-1}"

echo "Training EventMemAgent"
echo "  base model:  $BASE_MODEL_PATH"
echo "  train data:  $TRAIN_DATA"
if [[ -z "$VAL_DATA" ]]; then
  echo "  validation: off (no validation dataset or loader)"
else
  echo "  val data:    $VAL_DATA"
fi
echo "  output:      $TRAIN_OUTPUT_DIR"
echo "  GPUs:        $CUDA_VISIBLE_DEVICES"
echo "  GPU workers: $NUM_GPUS (rollout TP=$TENSOR_PARALLEL_SIZE)"
echo "  resume:      $RESUME_MODE${RESUME_FROM_PATH:+ from $RESUME_FROM_PATH}"

exec python -m verl.trainer.main_ppo \
  --config-path="${EVENTMEMAGENT_ROOT}/configs" \
  --config-name=train \
  "${hydra_overrides[@]}" \
  "$@"
