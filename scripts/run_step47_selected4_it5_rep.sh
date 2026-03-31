#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
export KMP_USE_SHM=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

BUDGET="${1:-252}"
if [[ "$BUDGET" != "252" && "$BUDGET" != "504" ]]; then
  echo "Usage: $0 [252|504]"
  exit 1
fi

run_cfg() {
  local cfg="$1"
  echo "[$(date '+%F %T')] START ${cfg}"
  PYTHONPATH=. /opt/anaconda3/envs/ml/bin/python scripts/run.py demo --config "$cfg"
  echo "[$(date '+%F %T')] DONE  ${cfg}"
}

run_cfg "configs/step47_selected4_sfamily/selected4_b${BUDGET}_lesr_iter_it5_sfamily_rep1.yaml"
run_cfg "configs/step47_selected4_sfamily/selected4_b${BUDGET}_lesr_iter_it5_global_rep1.yaml"

echo "[$(date '+%F %T')] Step47 rep complete for budget=${BUDGET}."
