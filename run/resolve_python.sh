#!/usr/bin/env bash
# Pick a Python that can import ogbench/jax. The named qflow env has mujoco-py
# 2.1 which crashes without mujoco210; fall back to the working QAM env (same jax).
set -euo pipefail
if [ -n "${QFLOW_PY:-}" ] && [ -x "${QFLOW_PY}" ]; then
  echo "$QFLOW_PY"
  exit 0
fi
for p in \
  /mnt/jhy/miniconda3/envs/qflow/bin/python \
  /mnt/zoe/conda-envs/qam/bin/python \
  /mnt/data/jmt/conda-envs/qam-qflow-v1/bin/python
do
  if [ -x "$p" ] && HOME=/mnt/zoe/home MUJOCO_GL=egl "$p" -c "import ogbench, jax, flax, wandb" >/dev/null 2>&1; then
    echo "$p"
    exit 0
  fi
done
echo "No working QFlow Python found (ogbench/jax import failed)." >&2
exit 1
