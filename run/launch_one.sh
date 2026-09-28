#!/usr/bin/env bash
# One QFlow region run on a currently idle GPU.
# Usage: bash run/launch_one.sh <gpu> <mask> <seed> <smoke|stage2|full>
set -euo pipefail

GPU="${1:?idle gpu index}"
MASK="${2:?guidance_mask_name}"
SEED="${3:?seed}"
MODE="${4:-full}"
ROOT=/mnt/zoe/projects/qflow
PY="$(bash "$ROOT/run/resolve_python.sh")"
LOGDIR="$ROOT/exp/logs"
mkdir -p "$LOGDIR"
SAFE_MASK="${MASK//_/-}"
LOG="$LOGDIR/qflow-${MODE}-${SAFE_MASK}-s${SEED}-gpu${GPU}.log"

export GPU
export HOME=/mnt/zoe/home
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export WANDB_DIR=/mnt/zoe/home/.wandb
export WANDB_ENTITY="${WANDB_ENTITY:-nightingale-314-}"
export WANDB_PROJECT="${WANDB_PROJECT:-qflow-region}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

LAMBDA="${QFLOW_LAMBDA:-0.2}"
FLOOR="${QFLOW_FLOOR:-0.25}"
ENV_NAME="${QFLOW_ENV:-cube-double-play-singletask-task2-v0}"

cd "$ROOT"

COMMON=(
  "$PY" main_region.py
  --agent=agents/qflow_region.py
  --env_name="$ENV_NAME"
  --agent.guidance_mask_name="$MASK"
  --agent.guidance_floor="$FLOOR"
  --agent.guidance_lambda="$LAMBDA"
  --agent.q_agg=mean
  --agent.use_time_embed=true
  --agent.time_embed_dim=16
  --agent.actor_loss_type=grad
  --seed="$SEED"
  --wandb_entity="$WANDB_ENTITY"
  --wandb_project="$WANDB_PROJECT"
)

echo "[$(date -Is)] QFLOW GPU=$GPU mask=$MASK seed=$SEED mode=$MODE py=$PY lambda=$LAMBDA entity=$WANDB_ENTITY" | tee -a "$LOG"

if [ "$MODE" = "smoke" ]; then
  bash "$ROOT/run/run_on_idle_gpu.sh" "${COMMON[@]}" \
    --run_group=qflow-region-smoke \
    --tags="QFLOW,${MASK},smoke,cube-double" \
    --offline_steps=200 \
    --online_steps=0 \
    --eval_interval=200 \
    --eval_episodes=2 \
    --log_interval=50 \
    --dump_interval=50 \
    --dump_samples=16 \
    --diag_interval=50 \
    --save_interval=0 \
    2>&1 | tee -a "$LOG"
  exit ${PIPESTATUS[0]}
fi

if [ "$MODE" = "stage2" ]; then
  bash "$ROOT/run/run_on_idle_gpu.sh" "${COMMON[@]}" \
    --run_group=qflow-stage2-diag \
    --tags="QFLOW,${MASK},stage2,diag,cube-double" \
    --offline_steps=1000000 \
    --online_steps=0 \
    --eval_interval=50000 \
    --eval_episodes=50 \
    --log_interval=5000 \
    --dump_interval=50000 \
    --dump_samples=64 \
    --diag_interval=5000 \
    --save_interval=1000000 \
    2>&1 | tee -a "$LOG"
  exit ${PIPESTATUS[0]}
fi

if [ "$MODE" != "full" ] && [ "$MODE" != "stage3" ]; then
  echo "MODE must be smoke|stage2|full|stage3, got: $MODE" >&2
  exit 2
fi

bash "$ROOT/run/run_on_idle_gpu.sh" "${COMMON[@]}" \
  --run_group=qflow-stage3-region \
  --tags="QFLOW,${MASK},stage3,cube-double" \
  --offline_steps=1000000 \
  --online_steps=0 \
  --eval_interval=50000 \
  --eval_episodes=50 \
  --log_interval=5000 \
  --dump_interval=0 \
  --diag_interval=5000 \
  --save_interval=1000000 \
  2>&1 | tee -a "$LOG"
exit ${PIPESTATUS[0]}
