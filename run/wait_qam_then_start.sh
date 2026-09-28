#!/usr/bin/env bash
# Wait for QAM Stage-3 full workers (GPUs 1/4/5) to finish, then start QFlow
# smoke + Stage 2/3. Never attaches to or kills QAM tmux sessions.
set -euo pipefail
ROOT=/mnt/zoe/projects/qflow
QAM=/mnt/zoe/projects/qam
GPUS=(1 4 5)
SHARDS=(0 1 2)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$ROOT/exp/logs"
LOG="$ROOT/exp/logs/qflow-wait-qam.log"

echo "[$(date -Is)] waiting for QAM Stage-3 workers qam-s3-g{1,4,5}" | tee -a "$LOG"

qam_done() {
  local i=0
  local g
  local n_tok
  for g in "${GPUS[@]}"; do
    local log="$QAM/exp/logs/stage3-worker-gpu${g}-shard${i}-full.log"
    if [ -f "$log" ] && grep -qE "worker GPU[= ]$g done" "$log"; then
      i=$((i + 1))
      continue
    fi
    # Fallback: all 9 seed-10001 runs wrote token.tk (workers may skip extra
    # seeds and exit without printing "done").
    n_tok=$(find "$QAM/exp/qam-reproduce/stage3-region" -name token.tk 2>/dev/null | wc -l)
    if [ "${n_tok:-0}" -ge 9 ]; then
      return 0
    fi
    return 1
  done
  return 0
}

while ! qam_done; do
  echo "[$(date -Is)] QAM Stage-3 still running; sleep 60s" | tee -a "$LOG"
  sleep 60
done
echo "[$(date -Is)] QAM Stage-3 workers reported done" | tee -a "$LOG"

for g in "${GPUS[@]}"; do
  bash "$ROOT/run/wait_idle_gpu.sh" "$g" 30 | tee -a "$LOG"
done

echo "[$(date -Is)] launching QFlow smoke then Stage 2/3" | tee -a "$LOG"
bash "$ROOT/run/launch_tmux.sh" all 2>&1 | tee -a "$LOG"
echo "[$(date -Is)] QFlow launcher returned $?" | tee -a "$LOG"
