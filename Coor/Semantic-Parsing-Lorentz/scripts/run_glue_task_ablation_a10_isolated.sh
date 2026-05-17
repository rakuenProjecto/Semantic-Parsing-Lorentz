#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home1/loss0109/projects/Semantic-Parsing-Lorentz/Coor/Semantic-Parsing-Lorentz"
CONDA_SH="/home1/loss0109/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="/home1/loss0109/miniconda3/envs/lorentz_text"
PYTHON_BIN="/home1/loss0109/miniconda3/envs/lorentz_text/bin/python"

cd "${PROJECT_DIR}"

HOST="$(hostname)"
TASK="${TASK:-mrpc}"
EPOCHS="${EPOCHS:-7}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

if [[ "${HOST}" == n001* ]]; then
  echo "[ERROR] This task ablation script must NOT be run on n001."
  echo "[ERROR] Current host: ${HOST}"
  exit 2
fi

if [[ "${HOST}" != n054* && "${ALLOW_NON_N054:-0}" != "1" ]]; then
  echo "[ERROR] Expected host n054, but current host is ${HOST}."
  echo "[ERROR] If this is another valid A10 node, run with ALLOW_NON_N054=1."
  exit 3
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

TRAIN_JSONL="data/real/glue/${TASK}/train.jsonl"
VAL_JSONL="data/real/glue/${TASK}/val.jsonl"
META_JSON="data/real/glue/${TASK}/meta.json"

if [[ ! -f "${TRAIN_JSONL}" || ! -f "${VAL_JSONL}" || ! -f "${META_JSON}" ]]; then
  echo "[ERROR] Missing JSONL files for task=${TASK}."
  echo "[INFO] Run: python scripts/prepare_glue_jsonl_task.py --task ${TASK}"
  exit 4
fi

NUM_TRAIN="$("${PYTHON_BIN}" - <<PY
import json
m=json.load(open("${META_JSON}", encoding="utf-8"))
print(m["num_train"])
PY
)"

NUM_VAL="$("${PYTHON_BIN}" - <<PY
import json
m=json.load(open("${META_JSON}", encoding="utf-8"))
print(m["num_val"])
PY
)"

RUN_TAG="glue_${TASK}_e${EPOCHS}_a10_isolated_${HOST}_${TS}"

LOG_ROOT="logs_task_isolated/${RUN_TAG}"
OUT_ROOT="outputs_task_isolated/${RUN_TAG}"
REPORT_ROOT="reports_task_isolated/${RUN_TAG}"

mkdir -p "${LOG_ROOT}" "${OUT_ROOT}" "${REPORT_ROOT}"

echo "=============================================="
echo "[INFO] Isolated GLUE task ablation runner"
echo "[INFO] HOST=${HOST}"
echo "[INFO] TASK=${TASK}"
echo "[INFO] EPOCHS=${EPOCHS}"
echo "[INFO] NUM_TRAIN=${NUM_TRAIN}"
echo "[INFO] NUM_VAL=${NUM_VAL}"
echo "[INFO] RUN_TAG=${RUN_TAG}"
echo "[INFO] LOG_ROOT=${LOG_ROOT}"
echo "[INFO] OUT_ROOT=${OUT_ROOT}"
echo "[INFO] REPORT_ROOT=${REPORT_ROOT}"
echo "[INFO] Python=$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
echo "=============================================="

nvidia-smi || true

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
  echo "[GPU] ${gpu_id}"
  echo "[TASK] ${TASK}"
  echo "[EPOCHS] ${EPOCHS}"
  echo "[CONFIG] ${config}"
  echo "[OUTPUT] ${output_dir}"
  echo "[LOG] ${log_path}"
  echo "[REPORTS] ${reports_dir}"
  echo "=============================================="

  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" train.py \
    --config "${config}" \
    --dataset_type jsonl \
    --train_jsonl "${TRAIN_JSONL}" \
    --val_jsonl "${VAL_JSONL}" \
    --complexity_mode heuristic \
    --epochs "${EPOCHS}" \
    --num_train_samples "${NUM_TRAIN}" \
    --num_val_samples "${NUM_VAL}" \
    --batch_size 8 \
    --output_dir "${output_dir}" \
    --empty_cache_on_epoch_end true \
    2>&1 | tee "${log_path}"

  local train_exit=${PIPESTATUS[0]}

  if [[ "${train_exit}" -ne 0 ]]; then
    echo "[ERROR] ${name} failed with exit code ${train_exit}"
    return "${train_exit}"
  fi

  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" scripts/summarize_latest_experiment.py \
    --output-dir "${output_dir}" \
    --log "${log_path}" \
    --cycle-id "${TASK}_${name}_a10_isolated" \
    --timestamp "${TS}" \
    --reports-dir "${reports_dir}"

  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" scripts/diagnose_experiment.py \
    --summary "${reports_dir}/latest_experiment_summary.json" \
    --cycle-id "${TASK}_${name}_a10_isolated" \
    --timestamp "${TS}" \
    --reports-dir "${reports_dir}"

  echo "=============================================="
  echo "[DONE] ${name}"
  echo "[OUTPUT] ${output_dir}"
  echo "[REPORT] ${reports_dir}/latest_experiment_summary.md"
  echo "=============================================="
}

run_one 0 main configs/bert_true_jacobian_anti_collapse.yaml &
PID0=$!

run_one 1 bert_no_jacobian_reg configs/ablations/bert_no_jacobian_reg.yaml &
PID1=$!

run_one 2 bert_no_curvature_aux configs/ablations/bert_no_curvature_aux.yaml &
PID2=$!

run_one 3 bert_no_aux_all configs/ablations/bert_no_aux_all.yaml &
PID3=$!

echo "=============================================="
echo "[INFO] Started ${TASK} ${EPOCHS}-epoch isolated task ablations"
echo "[INFO] main PID=${PID0}, GPU=0"
echo "[INFO] bert_no_jacobian_reg PID=${PID1}, GPU=1"
echo "[INFO] bert_no_curvature_aux PID=${PID2}, GPU=2"
echo "[INFO] bert_no_aux_all PID=${PID3}, GPU=3"
echo "=============================================="

FAIL=0
wait "${PID0}" || FAIL=1
wait "${PID1}" || FAIL=1
wait "${PID2}" || FAIL=1
wait "${PID3}" || FAIL=1

echo "=============================================="
if [[ "${FAIL}" -eq 0 ]]; then
  echo "[DONE] all ${TASK} ablations finished successfully"
else
  echo "[WARN] at least one ${TASK} ablation failed"
fi
echo "[INFO] RUN_TAG=${RUN_TAG}"
echo "[INFO] LOG_ROOT=${LOG_ROOT}"
echo "[INFO] OUT_ROOT=${OUT_ROOT}"
echo "[INFO] REPORT_ROOT=${REPORT_ROOT}"
echo "=============================================="

find "${OUT_ROOT}" -maxdepth 2 -type f -name "diagnostics_epoch_${EPOCHS}.json" -print | sort || true
find "${REPORT_ROOT}" -maxdepth 2 -type f -name "latest_experiment_summary.md" -print | sort || true

exit "${FAIL}"
