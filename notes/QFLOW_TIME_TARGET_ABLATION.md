# QFlow 时间目标消融

## 1. Action-chunking 版本

QAM 训练栈的新入口是：

- `../qam/main_qflow_chunk_region.py`
- agent 配置：`../qam/agents/qflow_chunk_region.py`
- CPU 测试：`../qam/stage3_target/test_qflow_chunk_region_cpu.py`

它使用 horizon=5 的扁平动作块，并把时间权重放进速度目标：

\[
v_{target}=u_t+w(t)\lambda^{-1}\nabla_xV(s,x_t,t).
\]

`NoQ` 仍以 \(u_t\) 监督整个时间区间。可选 schedule 为 `all`、`noq`、
`early50`、`late50`、`late25`、`remove_late25`。

GPU gate 建议先只跑 `All` 与 `NoQ`，同一 seed、200k offline steps。若
`All-NoQ` 小于跨 seed 波动，不应直接解释时间段；先检查 critic 梯度范数。
若 gate 通过，再跑其余四个 schedule，最后只对有差异的组合补 3 seeds。

命令模板（`GPU_ID` 必须由空闲卡检查得到）：

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} /mnt/zoe/conda-envs/qam/bin/python \
  main_qflow_chunk_region.py \
  --env_name=cube-triple-play-singletask-task2-v0 \
  --horizon_length=5 --offline_steps=200000 --online_steps=0 \
  --eval_interval=50000 --agent.target_schedule=all
```

## 2. 二维 learned/oracle 对照

入口：`2d_experiment/run_time_schedule_ablation.py`。

- `learned`：使用 time-conditioned critic 的 \(\nabla_xV(x_t,t)\)。
- `oracle`：对 \(R(\Phi_{t\rightarrow1}(x_t))\) 关于 \(x_t\) 求导，其中
  \(\Phi\) 是当前 flow 的后续积分。
- 两者共用同一数据、初始化、时间采样和 schedule。
- 主任务选 `swissroll`，因为其真实 reward 梯度连续；离散 `moons` 和
  `eight_gaussians` 不适合做 oracle 梯度基准。

完整快速屏幕：

```bash
cd /mnt/zoe/projects/qflow/2d_experiment
MPLCONFIGDIR=/mnt/tmp/mpl-qflow JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  /mnt/zoe/conda-envs/qam/bin/python run_time_schedule_ablation.py \
  --output=results/time_schedule_quick
```

当前 80-epoch CPU 初筛结果：

| gradient | NoQ | All | Early50 | Late50 | Late25 | RemoveLate25 |
|---|---:|---:|---:|---:|---:|---:|
| learned reward | 0.23207 | 0.23212 | 0.23209 | 0.23209 | 0.23208 | 0.23210 |
| oracle reward | 0.23207 | 0.23350 | 0.23294 | 0.23280 | 0.23258 | 0.23317 |

learned 与 oracle 梯度的平均余弦约 0.705，但 learned/oracle 范数比只有
0.0146。learned schedule 几乎没有拉开，而 oracle 已产生小幅排序。因此当前
更强的初步解释是 critic 梯度尺度不足。80 epochs 只用于筛查和验证管线，正式
结论需要更长训练和多 seed。

结果文件：`2d_experiment/results/time_schedule_quick/swissroll/metrics.csv` 和
`reward_comparison.png`。
