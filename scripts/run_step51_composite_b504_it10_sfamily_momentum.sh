#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
export KMP_USE_SHM=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "[$(date '+%F %T')] START configs/step51_composite_momentum_factor/composite_b504_lesr_iter_it10_sfamily_step51_momentum.yaml"
PYTHONPATH=. /opt/anaconda3/envs/ml/bin/python scripts/run.py demo \
  --config configs/step51_composite_momentum_factor/composite_b504_lesr_iter_it10_sfamily_step51_momentum.yaml

echo "[$(date '+%F %T')] DONE  configs/step51_composite_momentum_factor/composite_b504_lesr_iter_it10_sfamily_step51_momentum.yaml"
