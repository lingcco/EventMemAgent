#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/common.sh"
input_json="$(resolve_repo_path "${INPUT_JSON:-data/raw/input.json}")"
video_dir="$(resolve_repo_path "${VIDEO_DIR:-data/raw/videos}")"
output_dir="$(resolve_repo_path "${OUTPUT_DIR:-data/train}")"
embedding="$(resolve_repo_path "${EMBEDDING_MODEL_PATH:-checkpoints/pretrained/Qwen3-Embedding-0.6B}")"
require_file "$input_json" "input annotations"
require_path "$video_dir" "video directory"
require_path "$embedding" "embedding model"
require_http_health "${MEMORY_API_BASE:-http://localhost:8000/v1}/models" "4B caption service"
exec python -m eventmemagent.memory.preprocess \
  --input_json "$input_json" --video_dir "$video_dir" --output_dir "$output_dir" \
  --output_filename "${OUTPUT_FILENAME:-eventmemagent_train.parquet}" \
  --api_base "${MEMORY_API_BASE:-http://localhost:8000/v1}" \
  --api_key "${MEMORY_API_KEY:-EMPTY}" \
  --model_name "${MEMORY_CAPTION_MODEL_NAME:-Qwen3-VL-4B-Instruct}" \
  --embedding_model "$embedding" --sample_fps 1.0 --change_threshold 0.2 \
  --num_workers "${PREPROCESS_WORKERS:-4}" --batch_size "${PREPROCESS_BATCH_SIZE:-50}" "$@"
