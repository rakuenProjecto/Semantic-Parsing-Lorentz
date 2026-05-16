#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lorentz_text

cd "${PROJECT_DIR}"
mkdir -p logs outputs

CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/bert_true_jacobian_anti_collapse.yaml \
  --no_dummy_encoder \
  --encoder_name bert-base-uncased \
  --batch_size 2 \
  --epochs 3 \
  --num_train_samples 1024 \
  --num_val_samples 256 \
  --output_dir outputs/bert_true_jacobian_anti_collapse_v2_1gpu_bs2_e3_n1024 \
  2>&1 | tee logs/bert_true_jacobian_anti_collapse_v2_1gpu_bs2_e3_n1024.log
