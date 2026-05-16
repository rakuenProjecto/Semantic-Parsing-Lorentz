#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

mkdir -p logs reports/history
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate lorentz_text
fi

python scripts/autonomous_experiment_loop.py "$@" \
  2>&1 | tee "logs/autonomous_loop_master_${TIMESTAMP}.log"
