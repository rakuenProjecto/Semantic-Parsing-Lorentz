#!/usr/bin/env bash
set -euo pipefail

python train.py \
  --config configs/tiny_true_jacobian_smoke.yaml
