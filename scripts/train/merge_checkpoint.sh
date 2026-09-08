#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/common.sh"

TRAIN_OUTPUT_DIR="$(resolve_repo_path "${TRAIN_OUTPUT_DIR:-runs/train/eventmemagent_qwen3vl_8b}")"
TARGET_DIR="$(resolve_repo_path "${2:-${MODEL_PATH:-checkpoints/released/EventMemAgent-8B}}")"

if [[ -n "${1:-}" ]]; then
  ACTOR_DIR="$(resolve_repo_path "$1")"
else
  tracker="${TRAIN_OUTPUT_DIR}/latest_checkpointed_iteration.txt"
  require_file "$tracker" "latest-checkpoint tracker"
  step="$(tr -d '[:space:]' < "$tracker")"
  if [[ ! "$step" =~ ^[0-9]+$ ]]; then
    echo "Invalid checkpoint step in ${tracker}: ${step}" >&2
    exit 2
  fi
  ACTOR_DIR="${TRAIN_OUTPUT_DIR}/global_step_${step}/actor"
fi

require_path "$ACTOR_DIR" "verl actor checkpoint"
require_path "$ACTOR_DIR/huggingface" "checkpoint Hugging Face metadata"

if [[ -d "$TARGET_DIR" ]] && [[ -n "$(find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Target directory is not empty: ${TARGET_DIR}" >&2
  echo "Choose a new target directory. Existing releases are never overwritten automatically." >&2
  exit 2
fi
mkdir -p "$TARGET_DIR"

args=(
  merge
  --backend fsdp
  --local_dir "$ACTOR_DIR"
  --target_dir "$TARGET_DIR"
  --use_cpu_initialization
)

if [[ "${UPLOAD_TO_HF:-0}" == "1" ]]; then
  if [[ -z "${HF_MODEL_REPO:-}" || "$HF_MODEL_REPO" == YOUR_ORG/* ]]; then
    echo "Set HF_MODEL_REPO to your destination before requesting upload." >&2
    exit 2
  fi
  args+=(--hf_upload_path "$HF_MODEL_REPO")
  if [[ "${HF_PRIVATE:-0}" == "1" ]]; then
    args+=(--private)
  fi
fi

echo "Converting checkpoint"
echo "  source: $ACTOR_DIR"
echo "  target: $TARGET_DIR"
if [[ "${UPLOAD_TO_HF:-0}" == "1" ]]; then
  echo "  upload: https://huggingface.co/${HF_MODEL_REPO}"
fi

exec python -m verl.model_merger "${args[@]}"
