#!/usr/bin/env bash
# Run a command on one currently idle GPU. Refuse if that GPU is occupied.
# Usage: GPU=1 bash run/run_on_idle_gpu.sh python main_region.py ...
set -euo pipefail
ROOT=/mnt/zoe/projects/qflow

if [ -z "${GPU:-}" ]; then
  echo "Set GPU to an idle index first, e.g. GPU=1" >&2
  exit 2
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
procs=$(nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name --format=csv,noheader)
if [ -n "$procs" ] || [ "${used:-0}" -ge 200 ]; then
  echo "Refusing to run: GPU $GPU is occupied (used=${used} MiB)." >&2
  echo "$procs" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
echo "[guard] GPU=$GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES used=${used}MiB"
cd "$ROOT"
exec "$@"
