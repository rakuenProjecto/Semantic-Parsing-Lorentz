#!/usr/bin/env bash
set -euo pipefail

MODE="auto"
CYCLE_ID="${CYCLE_ID:-manual}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --cycle-id)
      CYCLE_ID="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

mkdir -p logs outputs reports/history
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/cycle_${MODE}_${CYCLE_ID}_${TIMESTAMP}.log"

run_train() {
  local mode="$1"
  local output_dir="outputs/cycle_${mode}_${CYCLE_ID}_${TIMESTAMP}"
  local batch_size="${BATCH_SIZE:-2}"
  if [[ -f reports/auto_batch_probe.json ]]; then
    batch_size="$(python - <<'PY'
import json
from pathlib import Path
p = Path("reports/auto_batch_probe.json")
print(json.loads(p.read_text()).get("safe_batch_size") or 2)
PY
)"
  fi

  case "${mode}" in
    debug)
      python train.py \
        --config configs/bert_true_jacobian_anti_collapse_debug.yaml \
        --epochs 1 \
        --num_train_samples 64 \
        --num_val_samples 32 \
        --batch_size 2 \
        --output_dir "${output_dir}"
      ;;
    short)
      python train.py \
        --config configs/bert_true_jacobian_anti_collapse.yaml \
        --epochs 1 \
        --num_train_samples 512 \
        --num_val_samples 128 \
        --batch_size "${batch_size}" \
        --output_dir "${output_dir}" \
        --empty_cache_on_epoch_end true
      ;;
    full)
      python train.py \
        --config configs/bert_true_jacobian_anti_collapse.yaml \
        --epochs 3 \
        --num_train_samples 1024 \
        --num_val_samples 256 \
        --batch_size "${batch_size}" \
        --output_dir "${output_dir}" \
        --empty_cache_on_epoch_end true
      ;;
    *)
      echo "unknown train mode: ${mode}" >&2
      exit 2
      ;;
  esac

  python scripts/summarize_latest_experiment.py \
    --output-dir "${output_dir}" \
    --log "${LOG_PATH}" \
    --cycle-id "${CYCLE_ID}" \
    --timestamp "${TIMESTAMP}"
  python scripts/diagnose_experiment.py --cycle-id "${CYCLE_ID}" --timestamp "${TIMESTAMP}"
}

{
  echo "cycle_id=${CYCLE_ID} mode=${MODE} timestamp=${TIMESTAMP}"
  if [[ "${MODE}" == "auto" ]]; then
    python scripts/gpu_probe.py
    python scripts/auto_batch_probe.py --config configs/bert_true_jacobian_anti_collapse_debug.yaml
    python -m pytest tests
    python train.py --config configs/tiny_true_jacobian_smoke.yaml
    run_train debug
    run_train short
  else
    run_train "${MODE}"
  fi
} 2>&1 | tee "${LOG_PATH}"
