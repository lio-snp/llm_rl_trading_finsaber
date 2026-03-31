#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
export KMP_USE_SHM=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "[$(date '+%F %T')] START configs/step48_delta_joint_selection/selected4_b504_lesr_iter_it10_sfamily_delta.yaml"
PYTHONPATH=. /opt/anaconda3/envs/ml/bin/python scripts/run.py demo \
  --config configs/step48_delta_joint_selection/selected4_b504_lesr_iter_it10_sfamily_delta.yaml

echo "[$(date '+%F %T')] DONE  configs/step48_delta_joint_selection/selected4_b504_lesr_iter_it10_sfamily_delta.yaml"
