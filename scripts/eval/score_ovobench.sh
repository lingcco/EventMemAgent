#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/common.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 EVAL_RESULTS_JSON [OVOBENCH_METADATA_JSON]" >&2
  exit 2
fi

eval_results="$(resolve_repo_path "$1")"
metadata="$(resolve_repo_path "${2:-${OVOBENCH_METADATA:-data/eval/ovobench/ovo_bench_new.json}}")"
require_file "$eval_results" "evaluation results"
require_file "$metadata" "OVO-Bench metadata"

python -m eventmemagent.evaluation.score_ovobench \
  --eval-results "$eval_results" \
  --ovo-bench-json "$metadata"
