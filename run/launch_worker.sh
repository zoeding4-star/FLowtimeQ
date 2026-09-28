#!/usr/bin/env bash
# Sequential QFlow jobs on ONE allowed idle GPU.
# Usage: bash run/launch_worker.sh <gpu> <shard_index> <n_shards> <smoke|stage2|stage3>
set -euo pipefail

GPU="${1:?idle gpu index}"
SHARD="${2:?shard index}"
NSHARD="${3:-2}"
MODE="${4:-stage3}"
ROOT=/mnt/zoe/projects/qflow
PY="$(bash "$ROOT/run/resolve_python.sh")"
LOGDIR="$ROOT/exp/logs"
mkdir -p "$LOGDIR"
MASTER_LOG="$LOGDIR/qflow-worker-gpu${GPU}-shard${SHARD}-${MODE}.log"

case "$GPU" in
  1|4|5) ;;
  *)
    echo "GPU $GPU is not in the allowed idle set 1,4,5. Refusing." >&2
    exit 2
    ;;
esac

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
echo "[$(date -Is)] worker GPU=$GPU shard=$SHARD/$NSHARD mode=$MODE py=$PY" | tee -a "$MASTER_LOG"

if [ "$MODE" = "stage2" ]; then
  mapfile -t JOBS < <("$PY" "$ROOT/stage3/jobs.py" --stage stage2)
else
  JOB_ARGS=(--stage "$MODE" --shard "$SHARD" --n-shards "$NSHARD")
  mapfile -t JOBS < <("$PY" "$ROOT/stage3/jobs.py" "${JOB_ARGS[@]}")
fi
echo "[$(date -Is)] ${#JOBS[@]} jobs: ${JOBS[*]}" | tee -a "$MASTER_LOG"

fail=0
for job in "${JOBS[@]}"; do
  # shellcheck disable=SC2086
  set -- $job
  MASK="$1"
  SEED="$2"
  KIND="${3:-$MODE}"
  echo "[$(date -Is)] START mask=$MASK seed=$SEED kind=$KIND on GPU $GPU" | tee -a "$MASTER_LOG"
  bash "$ROOT/run/wait_idle_gpu.sh" "$GPU" 30 | tee -a "$MASTER_LOG"
  rc=0
  bash "$ROOT/run/launch_one.sh" "$GPU" "$MASK" "$SEED" "$KIND" || rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "[$(date -Is)] OK mask=$MASK seed=$SEED" | tee -a "$MASTER_LOG"
  else
    echo "[$(date -Is)] FAIL mask=$MASK seed=$SEED (exit $rc). Continuing remaining jobs." | tee -a "$MASTER_LOG"
    fail=1
  fi
done

echo "[$(date -Is)] worker GPU=$GPU done fail=$fail" | tee -a "$MASTER_LOG"
exit "$fail"
