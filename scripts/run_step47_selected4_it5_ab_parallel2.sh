#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

export KMP_USE_SHM=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Force direct network path for API calls in this batch (avoid local VPN proxy leakage).
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="*"
export no_proxy="*"

TS="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="$ROOT_DIR/runs/_batch_logs/step47_it5_parallel2_$TS"
mkdir -p "$LOG_DIR"

CFG_1="configs/step47_selected4_sfamily/selected4_b252_lesr_iter_it5_global.yaml"
CFG_2="configs/step47_selected4_sfamily/selected4_b252_lesr_iter_it5_sfamily.yaml"
CFG_3="configs/step47_selected4_sfamily/selected4_b504_lesr_iter_it5_global.yaml"
CFG_4="configs/step47_selected4_sfamily/selected4_b504_lesr_iter_it5_sfamily.yaml"

echo "[$(date '+%F %T')] Batch start (parallel=2)"
echo "[$(date '+%F %T')] Logs: $LOG_DIR"

RUN_PID=""

run_one_bg() {
  local cfg="$1"
  local tag="$2"
  (
    echo "[$(date '+%F %T')] START $cfg"
    PYTHONPATH=. /opt/anaconda3/envs/ml/bin/python scripts/run.py demo --config "$cfg"
    rc=$?
    echo "[$(date '+%F %T')] END rc=$rc $cfg"
    exit "$rc"
  ) > "$LOG_DIR/$tag.log" 2>&1 &
  RUN_PID="$!"
}

wait_pair() {
  local p1="$1"
  local p2="$2"
  local rc=0

  wait "$p1" || rc=1
  wait "$p2" || rc=1

  return "$rc"
}

overall_rc=0

echo "[$(date '+%F %T')] Wave-1 launch: b252 global + b252 sfamily"
run_one_bg "$CFG_1" "b252_global"
pid1="$RUN_PID"
run_one_bg "$CFG_2" "b252_sfamily"
pid2="$RUN_PID"
echo "[$(date '+%F %T')] Wave-1 pids: $pid1 $pid2"
wait_pair "$pid1" "$pid2" || overall_rc=1

echo "[$(date '+%F %T')] Wave-2 launch: b504 global + b504 sfamily"
run_one_bg "$CFG_3" "b504_global"
pid3="$RUN_PID"
run_one_bg "$CFG_4" "b504_sfamily"
pid4="$RUN_PID"
echo "[$(date '+%F %T')] Wave-2 pids: $pid3 $pid4"
wait_pair "$pid3" "$pid4" || overall_rc=1

echo "[$(date '+%F %T')] Batch done rc=$overall_rc"
echo "[$(date '+%F %T')] Logs: $LOG_DIR"
exit "$overall_rc"
