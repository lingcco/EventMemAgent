#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/common.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 EVAL_RESULTS_JSON [MODEL_NAME]" >&2
  exit 2
fi

eval_results="$(resolve_repo_path "$1")"
require_file "$eval_results" "evaluation results"

questions="$(resolve_repo_path "${STREAMINGBENCH_DATA:-data/eval/streamingbench/questions_real.json}")"
model_name="${MODEL_NAME:-EventMemAgent}"
shift
if [[ $# -gt 0 && "$1" != --* ]]; then
  model_name="$1"
  shift
fi
python -m eventmemagent.evaluation.score_streamingbench \
  --eval-results "$eval_results" \
  --model-name "$model_name" \
  --questions-json "$questions" "$@"
