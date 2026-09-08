#!/usr/bin/env bash

set -euo pipefail

EVENTMEMAGENT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVENTMEMAGENT_ROOT="$(cd "${EVENTMEMAGENT_SCRIPT_DIR}/../.." && pwd)"

if [[ -f "${EVENTMEMAGENT_ROOT}/.env" ]]; then
  while IFS= read -r dotenv_line || [[ -n "$dotenv_line" ]]; do
    dotenv_line="${dotenv_line#"${dotenv_line%%[![:space:]]*}"}"
    dotenv_line="${dotenv_line%"${dotenv_line##*[![:space:]]}"}"
    [[ -z "$dotenv_line" || "$dotenv_line" == \#* ]] && continue
    if [[ "$dotenv_line" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
      dotenv_key="${BASH_REMATCH[1]}"
      dotenv_value="${BASH_REMATCH[2]}"
      if [[ -z "${!dotenv_key+x}" ]]; then
        if [[ "$dotenv_value" =~ ^\"(.*)\"$ || "$dotenv_value" =~ ^\'(.*)\'$ ]]; then
          dotenv_value="${BASH_REMATCH[1]}"
        fi
        export "${dotenv_key}=${dotenv_value}"
      fi
    else
      echo "Ignoring an invalid .env line (expected KEY=VALUE)." >&2
    fi
  done < "${EVENTMEMAGENT_ROOT}/.env"
fi

resolve_repo_path() {
  local value="$1"
  if [[ "$value" = /* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s\n' "${EVENTMEMAGENT_ROOT}/${value#./}"
  fi
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "Missing ${label}: ${path}" >&2
    echo "Update .env or see the corresponding README under data/ or checkpoints/." >&2
    exit 2
  fi
}

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    echo "Missing ${label}: ${path}" >&2
    echo "Update .env or see the corresponding README under data/ or checkpoints/." >&2
    exit 2
  fi
}

require_positive_integer() {
  local value="$1"
  local label="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "${label} must be a positive integer, got: ${value}" >&2
    exit 2
  fi
}

count_csv_items() {
  local value="$1"
  local -a items
  IFS=',' read -r -a items <<< "$value"
  printf '%s\n' "${#items[@]}"
}

require_http_health() {
  local url="$1"
  local label="$2"
  if ! curl --fail --silent --show-error --max-time 5 "$url" >/dev/null; then
    echo "${label} is not healthy: ${url}" >&2
    echo "Start the service first, update its URL in .env, or set SKIP_TOOL_SERVER_CHECK=1." >&2
    exit 2
  fi
}

configure_eval_devices() {
  export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
  export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${EVAL_CUDA_VISIBLE_DEVICES:-0}}"
  export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${EVAL_TENSOR_PARALLEL_SIZE:-1}}"
  export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-${EVAL_GPU_MEMORY_UTILIZATION:-0.8}}"
  export MAX_MODEL_LEN="${MAX_MODEL_LEN:-${EVAL_MAX_MODEL_LEN:-32768}}"
  require_positive_integer "$TENSOR_PARALLEL_SIZE" "evaluation tensor parallel size"
  if (( TENSOR_PARALLEL_SIZE > $(count_csv_items "$CUDA_VISIBLE_DEVICES") )); then
    echo "Evaluation tensor parallel size exceeds the number of visible GPUs." >&2
    exit 2
  fi
}

cd "$EVENTMEMAGENT_ROOT"
