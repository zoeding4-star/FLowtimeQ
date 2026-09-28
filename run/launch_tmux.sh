#!/usr/bin/env bash
# Start QFlow smoke, then Stage 2 + Stage 3 in parallel on GPUs 1/4/5.
# Does not start until those GPUs are idle. Does not touch 0/2/3/6/7.
#
#   bash run/launch_tmux.sh          # smoke then stage2+stage3 (this process waits)
#   bash run/launch_tmux.sh smoke
#   bash run/launch_tmux.sh after-smoke
set -euo pipefail

PHASE="${1:-all}"
ROOT=/mnt/zoe/projects/qflow
GPUS=(1 4 5)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd "$ROOT"

idle_ok() {
  local gpu="$1"
  local used procs
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  procs=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | grep -c '[0-9]' || true)
  [ "${procs:-0}" -eq 0 ] && [ "${used:-0}" -lt 200 ]
}

worker_log() {
  local gpu="$1"
  local shard="$2"
  local mode="$3"
  echo "$ROOT/exp/logs/qflow-worker-gpu${gpu}-shard${shard}-${mode}.log"
}

rotate_worker_log() {
  local log="$1"
  if [ -f "$log" ]; then
    mv "$log" "${log}.prev.$(date +%Y%m%dT%H%M%S)"
  fi
}

start_workers() {
  local mode="$1"
  local prefix="$2"
  local nshard="$3"
  local i=0
  for g in "${GPUS[@]}"; do
    local sess="${prefix}-g${g}"
    if tmux has-session -t "$sess" 2>/dev/null; then
      if [ "${QFLOW_REPLACE_TMUX:-0}" = "1" ]; then
        echo "replacing leftover tmux $sess"
        tmux kill-session -t "$sess"
      else
        echo "tmux session $sess already exists; not clobbering it." >&2
        exit 1
      fi
    fi
    rotate_worker_log "$(worker_log "$g" "$i" "$mode")"
    tmux new-session -d -s "$sess" \
      "cd $ROOT && bash $ROOT/run/launch_worker.sh $g $i $nshard $mode; echo EXIT:\$?; exec bash"
    echo "started tmux $sess  (attach: tmux attach -t $sess)"
    i=$((i + 1))
  done
}

last_done_line() {
  local gpu="$1"
  local log="$2"
  grep -E "worker GPU=$gpu done fail=" "$log" 2>/dev/null | tail -n 1 || true
}

wait_workers() {
  local mode="$1"
  echo "[$(date -Is)] waiting for $mode workers"
  while true; do
    local alive=0
    local i=0
    for g in "${GPUS[@]}"; do
      local log
      log="$(worker_log "$g" "$i" "$mode")"
      if [ ! -f "$log" ] || [ -z "$(last_done_line "$g" "$log")" ]; then
        alive=1
      fi
      i=$((i + 1))
    done
    if [ "$alive" -eq 0 ]; then
      echo "[$(date -Is)] all $mode workers reported done"
      return 0
    fi
    sleep 20
  done
}

any_worker_failed() {
  local mode="$1"
  local i=0
  local g log line
  for g in "${GPUS[@]}"; do
    log="$(worker_log "$g" "$i" "$mode")"
    line="$(last_done_line "$g" "$log")"
    if [[ "$line" == *"fail=1" ]]; then
      return 0
    fi
    i=$((i + 1))
  done
  return 1
}

echo "[$(date -Is)] checking allowed GPUs ${GPUS[*]}"
for g in "${GPUS[@]}"; do
  if ! idle_ok "$g"; then
    echo "GPU $g is not idle. Waiting rather than sharing/stealing."
    bash "$ROOT/run/wait_idle_gpu.sh" "$g" 30
  fi
done

if [ "$PHASE" = "smoke" ] || [ "$PHASE" = "all" ]; then
  start_workers smoke qflow-smoke 3
fi

if [ "$PHASE" = "all" ]; then
  wait_workers smoke
  if any_worker_failed smoke; then
    echo "A smoke worker reported fail=1. Not launching full Stage 2/3." >&2
    exit 1
  fi
  PYTHONPATH="$ROOT" "$(bash "$ROOT/run/resolve_python.sh")" "$ROOT/stage3/check_smoke.py"
  PHASE=after-smoke
fi

if [ "$PHASE" = "after-smoke" ]; then
  # GPU 1: Stage-2 diagnostics (All). GPU 4/5: Stage-3 ablation shards.
  if tmux has-session -t qflow-s2-g1 2>/dev/null; then
    echo "tmux qflow-s2-g1 already exists; not clobbering." >&2
    exit 1
  fi
  tmux new-session -d -s qflow-s2-g1 \
    "cd $ROOT && bash $ROOT/run/launch_worker.sh 1 0 1 stage2; echo EXIT:\$?; exec bash"
  echo "started tmux qflow-s2-g1"

  for spec in "4 0" "5 1"; do
    set -- $spec
    g="$1"; shard="$2"
    sess="qflow-s3-g${g}"
    if tmux has-session -t "$sess" 2>/dev/null; then
      echo "tmux $sess already exists; not clobbering." >&2
      exit 1
    fi
    tmux new-session -d -s "$sess" \
      "cd $ROOT && bash $ROOT/run/launch_worker.sh $g $shard 2 stage3; echo EXIT:\$?; exec bash"
    echo "started tmux $sess"
  done
  echo "[$(date -Is)] Stage 2 on GPU 1, Stage 3 on GPUs 4 and 5"
elif [ "$PHASE" = "smoke" ]; then
  echo "smoke workers started. Attach with: tmux attach -t qflow-smoke-g1"
else
  echo "phase must be all|smoke|after-smoke" >&2
  exit 2
fi
