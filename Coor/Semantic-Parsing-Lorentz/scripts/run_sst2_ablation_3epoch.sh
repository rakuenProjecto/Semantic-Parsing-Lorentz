#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

run_ablation() {
  local name="$1"
  local config="$2"
  local output_dir="outputs/ablations/${name}_${TIMESTAMP}"
  local log_path="logs/ablations/${name}_${TIMESTAMP}.log"
  local reports_dir="reports/ablations/${name}_${TIMESTAMP}"

  mkdir -p "$(dirname "${log_path}")"

  python train.py \
    --config "${config}" \
    --dataset_type jsonl \
    --train_jsonl data/real/sst2_full/train.jsonl \
    --val_jsonl data/real/sst2_full/val.jsonl \
    --num_train_samples 67349 \
    --num_val_samples 872 \
    --batch_size 8 \
    --epochs 3 \
    --output_dir "${output_dir}" \
    2>&1 | tee "${log_path}"

  python scripts/summarize_latest_experiment.py \
    --output-dir "${output_dir}" \
    --log "${log_path}" \
    --cycle-id "${name}" \
    --timestamp "${TIMESTAMP}" \
    --reports-dir "${reports_dir}"

  python scripts/diagnose_experiment.py \
    --summary "${reports_dir}/latest_experiment_summary.json" \
    --cycle-id "${name}" \
    --timestamp "${TIMESTAMP}" \
    --reports-dir "${reports_dir}"
}

run_ablation "bert_no_jacobian_reg" "configs/ablations/bert_no_jacobian_reg.yaml"
run_ablation "bert_no_curvature_aux" "configs/ablations/bert_no_curvature_aux.yaml"
run_ablation "bert_no_aux_all" "configs/ablations/bert_no_aux_all.yaml"
