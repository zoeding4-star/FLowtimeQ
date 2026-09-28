# QFlow Stage 2 / 3（cube-double）

日期：2026-09-19  
代码目录：`/mnt/zoe/projects/qflow`（**不改** `agents/qflow.py` / `main.py` / `agents/qflow_qamsetting.py`）  
任务：`cube-double-play-singletask-task2-v0`（与 QAM Stage 2/3 同一环境；这里是 **QFlow standard setting**，无 action chunking）  
W&amp;B：entity `nightingale-314-`，project `qflow-region`  
Python：优先 `/mnt/jhy/miniconda3/envs/qflow`，但该环境的 `mujoco-py` 缺 `mujoco210` 会在 import ogbench 时崩。启动脚本会自动退到 `/mnt/zoe/conda-envs/qam`（jax/flax/ogbench 版本相同，且已能跑 cube）。

QAM 的 Stage 3（`qam-s3-g1/g4/g5`）按原计划跑完，**本目录的 waiter 只等、不杀、不占卡**。

## 问题

QAM 0/1 mask 把「这段 Q 有没有用」和「`actor_fast` 还训不训」缠在一起。QFlow 只有一条场，BC 速度始终在：

$$
v^\star(x_t,t)=u_t + w(t)\,\beta\nabla_{x_t}Q_i(s,x_t,t)
$$

$w(t)$ 只缩放 inner-Q 引导。除 No-Q 外，$w_k>0$ 且 $\mathrm{mean}(w)=1$。

QFlow 的 $p_t$ 对应物是 $\nabla_{x_t}Q_i$，不是 ODE adjoint。Stage 2 同时记：

- **interpolant** $x_t=(1-t)x_0+tx_1$（训练用，主 $p_t$）
- **Euler policy path**（eval 真正走的轨迹）

看 early 的 $\|\nabla Q_i\|$ 会不会仍像 QAM 的 $\|p_t\|$ 那样塌掉。

## Stage 1

不做完整 QFlow 复现（之前已经跑过 antmaze）。只做 **smoke**：7 个 mask，200 step，确认 weight / loss / dump。

## Stage 3 矩阵（floor $=0.25$，再归一化到 mean 1）

| ID | 方法 | 含义 |
| --- | --- | --- |
| M0 | No-Q | $w=0$，纯 BC |
| M1 | All | $w=1$ |
| M2 | Early-boost | 地板 + early 更高 |
| M3 | Middle-boost | 地板 + middle 更高 |
| M4 | Late-boost | 地板 + late 更高 |
| M5 | Increasing | $w\propto \mathrm{floor}+(1-\mathrm{floor})t$ |
| M6 | Decreasing | $w\propto \mathrm{floor}+(1-\mathrm{floor})(1-t)$ |

1 seed（10001）。超参对齐已跑通的 QFlow：`guidance_lambda=0.2`，`use_time_embed=true`，`time_embed_dim=16`，`actor_loss_type=grad`，`q_agg=mean`，1M offline、0 online。

## 怎么跑

```bash
# CPU 单测 + 画权重图
PYTHONPATH=/mnt/zoe/projects/qflow \
  /mnt/zoe/conda-envs/qam/bin/python /mnt/zoe/projects/qflow/stage3/test_guidance_weights.py
PYTHONPATH=/mnt/zoe/projects/qflow \
  /mnt/zoe/conda-envs/qam/bin/python /mnt/zoe/projects/qflow/stage3/plot_weights.py

# 等 QAM Stage 3 全部结束后自动：smoke → Stage2(GPU1) + Stage3(GPU4/5)
# 已由 waiter tmux `qflow-wait` 挂起（不占 GPU）
tmux attach -t qflow-wait
```

允许的 GPU：**只 1 / 4 / 5**。0/2/3/6/7 上有别人的 Blender。

## 文件

| 路径 | 作用 |
| --- | --- |
| `stage3/guidance_weights.py` | $w(t)$ |
| `agents/qflow_region.py` | 加权 guidance + `diagnostic_snapshot` |
| `main_region.py` | 训练循环 + diag dump |
| `run/wait_qam_then_start.sh` | 等 QAM 结束再启动 |
| `run/launch_tmux.sh` | smoke 然后 Stage2/3 并行 |
