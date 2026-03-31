#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
export KMP_USE_SHM=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

run_cfg() {
  local cfg="$1"
  echo "[$(date '+%F %T')] START ${cfg}"
  PYTHONPATH=. /opt/anaconda3/envs/ml/bin/python scripts/run.py demo --config "$cfg"
  echo "[$(date '+%F %T')] DONE  ${cfg}"
}

run_cfg "configs/step47_backend_control/index_selected4_b252_it5_backend_custom_fixed.yaml"
run_cfg "configs/step47_backend_control/index_selected4_b252_it5_backend_sb3_fixed.yaml"

echo "[$(date '+%F %T')] Phase1 complete."
