#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home1/loss0109/projects/Semantic-Parsing-Lorentz/Coor/Semantic-Parsing-Lorentz"
CONDA_SH="/home1/loss0109/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="/home1/loss0109/miniconda3/envs/lorentz_text"
PYTHON_BIN="/home1/loss0109/miniconda3/envs/lorentz_text/bin/python"

cd "${PROJECT_DIR}"

HOST="$(hostname)"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

# n001 작업을 실수로 건드리지 않기 위한 안전장치
if [[ "${HOST}" == n001* ]]; then
  echo "[ERROR] This isolated A10 script must NOT be run on n001."
  echo "[ERROR] Current host: ${HOST}"
  echo "[ERROR] Move to n054 or another A10 node first."
  exit 2
fi

# n054 전용으로 강하게 제한하고 싶으면 여기서 통과.
# 다른 A10 노드에서 실행해야 할 경우 ALLOW_NON_N054=1 을 붙이면 됨.
if [[ "${HOST}" != n054* && "${ALLOW_NON_N054:-0}" != "1" ]]; then
  echo "[ERROR] Expected host n054, but current host is ${HOST}."
  echo "[ERROR] If this is another valid A10 node, run with ALLOW_NON_N054=1."
  exit 3
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

A10_ROOT_TAG="fresh_a10_isolated_${HOST}_${TS}"

LOG_ROOT="logs_a10_isolated/ablations/${A10_ROOT_TAG}"
OUT_ROOT="outputs_a10_isolated/ablations/${A10_ROOT_TAG}"
REPORT_ROOT="reports_a10_isolated/ablations/${A10_ROOT_TAG}"

mkdir -p "${LOG_ROOT}" "${OUT_ROOT}" "${REPORT_ROOT}"

echo "=============================================="
echo "[INFO] Isolated A10 ablation runner"
echo "[INFO] HOST=${HOST}"
echo "[INFO] TS=${TS}"
echo "[INFO] LOG_ROOT=${LOG_ROOT}"
echo "[INFO] OUT_ROOT=${OUT_ROOT}"
echo "[INFO] REPORT_ROOT=${REPORT_ROOT}"
echo "[INFO] Python=$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
echo "=============================================="

echo "===== GPU status before run ====="
nvidia-smi || true

echo "===== required files check ====="
for f in \
  train.py \
  configs/ablations/bert_no_jacobian_reg.yaml \
  configs/ablations/bert_no_curvature_aux.yaml \
  configs/ablations/bert_no_aux_all.yaml \
  data/real/sst2_full/train.jsonl \
  data/real/sst2_full/val.jsonl \
  scripts/summarize_latest_experiment.py \
  scripts/diagnose_experiment.py
do
  if [[ ! -f "${f}" ]]; then
    echo "[ERROR] Missing required file: ${f}"
    exit 4
  fi
  echo "[OK] ${f}"
done

# DRY_RUN=1이면 실제 학습 없이 실행될 명령만 출력
DRY_RUN="${DRY_RUN:-0}"

run_one() {
  local gpu_id="$1"
  local name="$2"
  local config="$3"

  local output_dir="${OUT_ROOT}/${name}"
  local log_path="${LOG_ROOT}/${name}.log"
  local reports_dir="${REPORT_ROOT}/${name}"

  mkdir -p "${reports_dir}"

  echo "=============================================="
  echo "[START] ${name}"
  echo "[HOST] ${HOST}"
  echo "[GPU] ${gpu_id}"
  echo "[CONFIG] ${config}"
  echo "[OUTPUT] ${output_dir}"
  echo "[LOG] ${log_path}"
  echo "[REPORTS] ${reports_dir}"
  echo "=============================================="

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY-RUN] CUDA_VISIBLE_DEVICES=${gpu_id} ${PYTHON_BIN} train.py \\"
    echo "  --config ${config} \\"
    echo "  --dataset_type jsonl \\"
    echo "  --train_jsonl data/real/sst2_full/train.jsonl \\"
    echo "  --val_jsonl data/real/sst2_full/val.jsonl \\"
    echo "  --complexity_mode heuristic \\"
    echo "  --epochs 3 \\"
    echo "  --num_train_samples 67349 \\"
    echo "  --num_val_samples 872 \\"
    echo "  --batch_size 8 \\"
    echo "  --output_dir ${output_dir} \\"
    echo "  --empty_cache_on_epoch_end true"
    return 0
  fi

  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" train.py \
    --config "${config}" \
    --dataset_type jsonl \
    --train_jsonl data/real/sst2_full/train.jsonl \
    --val_jsonl data/real/sst2_full/val.jsonl \
    --complexity_mode heuristic \
    --epochs 3 \
    --num_train_samples 67349 \
    --num_val_samples 872 \
    --batch_size 8 \
    --output_dir "${output_dir}" \
    --empty_cache_on_epoch_end true \
    2>&1 | tee "${log_path}"

  local train_exit=${PIPESTATUS[0]}

  if [[ "${train_exit}" -ne 0 ]]; then
    echo "[ERROR] ${name} train failed with exit code ${train_exit}"
    return "${train_exit}"
  fi

  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" scripts/summarize_latest_experiment.py \
    --output-dir "${output_dir}" \
    --log "${log_path}" \
    --cycle-id "${name}_a10_isolated" \
    --timestamp "${TS}" \
    --reports-dir "${reports_dir}"

  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" scripts/diagnose_experiment.py \
    --summary "${reports_dir}/latest_experiment_summary.json" \
    --cycle-id "${name}_a10_isolated" \
    --timestamp "${TS}" \
    --reports-dir "${reports_dir}"

  echo "=============================================="
  echo "[DONE] ${name}"
  echo "[OUTPUT] ${output_dir}"
  echo "[REPORT] ${reports_dir}/latest_experiment_summary.md"
  echo "=============================================="
}

run_one 0 bert_no_jacobian_reg configs/ablations/bert_no_jacobian_reg.yaml &
PID1=$!

run_one 1 bert_no_curvature_aux configs/ablations/bert_no_curvature_aux.yaml &
PID2=$!

run_one 2 bert_no_aux_all configs/ablations/bert_no_aux_all.yaml &
PID3=$!

echo "=============================================="
echo "[INFO] Started isolated A10 ablations"
echo "[INFO] HOST=${HOST}"
echo "[INFO] TS=${TS}"
echo "[INFO] bert_no_jacobian_reg PID=${PID1}, GPU=0"
echo "[INFO] bert_no_curvature_aux PID=${PID2}, GPU=1"
echo "[INFO] bert_no_aux_all PID=${PID3}, GPU=2"
echo "[INFO] GPU 3 is intentionally left free."
echo "=============================================="

FAIL=0

wait "${PID1}" || FAIL=1
wait "${PID2}" || FAIL=1
wait "${PID3}" || FAIL=1

echo "=============================================="
if [[ "${FAIL}" -eq 0 ]]; then
  echo "[DONE] all isolated A10 ablations finished successfully"
else
  echo "[WARN] at least one isolated A10 ablation failed"
fi
echo "[INFO] HOST=${HOST}"
echo "[INFO] TS=${TS}"
echo "[INFO] LOG_ROOT=${LOG_ROOT}"
echo "[INFO] OUT_ROOT=${OUT_ROOT}"
echo "[INFO] REPORT_ROOT=${REPORT_ROOT}"
echo "=============================================="

if [[ "${DRY_RUN}" != "1" ]]; then
  echo "===== generated summaries ====="
  find "${REPORT_ROOT}" -maxdepth 2 -type f -name "latest_experiment_summary.md" -print | sort

  echo "===== generated diagnostics ====="
  find "${OUT_ROOT}" -maxdepth 2 -type f -name "diagnostics_epoch_3.json" -print | sort
fi

exit "${FAIL}"
