#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home1/loss0109/projects/Semantic-Parsing-Lorentz/Coor/Semantic-Parsing-Lorentz"
CONDA_SH="/home1/loss0109/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="/home1/loss0109/miniconda3/envs/lorentz_text"
PYTHON_BIN="/home1/loss0109/miniconda3/envs/lorentz_text/bin/python"

cd "${PROJECT_DIR}"

HOST="$(hostname)"

if [[ "${HOST}" == n001* ]]; then
  echo "[ERROR] This script must NOT be run on n001."
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

# 이미 돌아가는 isolated ablation과 같은 timestamp 폴더를 가능하면 재사용
LATEST_ROOT="$(find logs_a10_isolated/ablations -maxdepth 1 -type d -name 'fresh_a10_isolated_*' 2>/dev/null | sort | tail -n 1 || true)"

if [[ -n "${LATEST_ROOT}" ]]; then
  RUN_TAG="$(basename "${LATEST_ROOT}")"
else
  RUN_TAG="fresh_a10_isolated_${HOST}_$(date +%Y%m%d_%H%M%S)"
fi

LOG_ROOT="logs_a10_isolated/ablations/${RUN_TAG}"
OUT_ROOT="outputs_a10_isolated/ablations/${RUN_TAG}"
REPORT_ROOT="reports_a10_isolated/ablations/${RUN_TAG}"

NAME="main"
GPU_ID=3
CONFIG="configs/bert_true_jacobian_anti_collapse.yaml"
OUTPUT_DIR="${OUT_ROOT}/${NAME}"
LOG_PATH="${LOG_ROOT}/${NAME}.log"
REPORTS_DIR="${REPORT_ROOT}/${NAME}"

mkdir -p "${LOG_ROOT}" "${OUT_ROOT}" "${REPORT_ROOT}" "${REPORTS_DIR}"

echo "=============================================="
echo "[INFO] Isolated A10 main runner"
echo "[INFO] HOST=${HOST}"
echo "[INFO] RUN_TAG=${RUN_TAG}"
echo "[INFO] GPU_ID=${GPU_ID}"
echo "[INFO] CONFIG=${CONFIG}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] LOG_PATH=${LOG_PATH}"
echo "[INFO] REPORTS_DIR=${REPORTS_DIR}"
echo "[INFO] Python=$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
echo "=============================================="

echo "===== GPU status before run ====="
nvidia-smi || true

echo "===== required files check ====="
for f in \
  train.py \
  "${CONFIG}" \
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

if pgrep -af "train.py" | grep -q "${OUTPUT_DIR}"; then
  echo "[ERROR] main isolated run already seems to be running for OUTPUT_DIR=${OUTPUT_DIR}"
  pgrep -af "train.py" | grep "${OUTPUT_DIR}" || true
  exit 5
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train.py \
  --config "${CONFIG}" \
  --dataset_type jsonl \
  --train_jsonl data/real/sst2_full/train.jsonl \
  --val_jsonl data/real/sst2_full/val.jsonl \
  --complexity_mode heuristic \
  --epochs 3 \
  --num_train_samples 67349 \
  --num_val_samples 872 \
  --batch_size 8 \
  --output_dir "${OUTPUT_DIR}" \
  --empty_cache_on_epoch_end true \
  2>&1 | tee "${LOG_PATH}"

TRAIN_EXIT=${PIPESTATUS[0]}

if [[ "${TRAIN_EXIT}" -ne 0 ]]; then
  echo "[ERROR] main train failed with exit code ${TRAIN_EXIT}"
  exit "${TRAIN_EXIT}"
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" scripts/summarize_latest_experiment.py \
  --output-dir "${OUTPUT_DIR}" \
  --log "${LOG_PATH}" \
  --cycle-id "main_a10_isolated_gpu3" \
  --timestamp "${RUN_TAG}" \
  --reports-dir "${REPORTS_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" scripts/diagnose_experiment.py \
  --summary "${REPORTS_DIR}/latest_experiment_summary.json" \
  --cycle-id "main_a10_isolated_gpu3" \
  --timestamp "${RUN_TAG}" \
  --reports-dir "${REPORTS_DIR}"

echo "=============================================="
echo "[DONE] isolated A10 main finished"
echo "[OUTPUT] ${OUTPUT_DIR}"
echo "[LOG] ${LOG_PATH}"
echo "[REPORT] ${REPORTS_DIR}/latest_experiment_summary.md"
echo "=============================================="
