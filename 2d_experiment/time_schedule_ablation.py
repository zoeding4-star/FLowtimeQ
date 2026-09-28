"""Fast 2D QFlow ablation for temporal schedules and gradient sources.

The oracle source differentiates the true terminal reward through the current
flow continuation from (x_t, t) to t=1.  It therefore estimates the same
continuation-value gradient that the learned time-conditioned critic targets.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax

from networks import (
    apply_mlp,
    critic_forward,
    init_mlp,
    integrate_flow,
    integrate_flow_from,
)


SCHEDULES = ("all", "noq", "early50", "late50", "late25", "remove_late25")
GRADIENT_SOURCES = ("learned", "oracle")


def schedule_weight(t, name):
    """Return an unnormalized target multiplier at continuous flow time t."""
    name = str(name).strip().lower().replace("-", "_")
    name = {"no_q": "noq", "early": "early50", "late": "late50"}.get(name, name)
    if name == "all":
        return jnp.ones_like(t)
    if name == "noq":
        return jnp.zeros_like(t)
    if name == "early50":
        return (t < 0.5).astype(t.dtype)
    if name == "late50":
        return (t >= 0.5).astype(t.dtype)
    if name == "late25":
        return (t >= 0.7).astype(t.dtype)
    if name == "remove_late25":
        return (t < 0.7).astype(t.dtype)
    raise ValueError(f"unknown schedule {name!r}; expected one of {SCHEDULES}")


def train_schedule_ablation(
    x_1_data,
    rewards_data,
    reward_fn_jnp,
    *,
    schedule="all",
    gradient_source="learned",
    epochs=80,
    bc_epochs=40,
    batch_size=256,
    hidden=64,
    num_layers=2,
    flow_steps=10,
    alpha=1.0,
    lr=3e-4,
    seed=0,
    num_evals=1024,
):
    """Train one compact QFlow variant and return parameters and diagnostics."""
    if schedule not in SCHEDULES:
        raise ValueError(f"unknown schedule {schedule!r}")
    if gradient_source not in GRADIENT_SOURCES:
        raise ValueError(f"unknown gradient source {gradient_source!r}")
    if not (0 <= bc_epochs <= epochs):
        raise ValueError("bc_epochs must be between zero and epochs")

    X1 = jnp.asarray(x_1_data, dtype=jnp.float32)
    R = jnp.asarray(rewards_data, dtype=jnp.float32).reshape(-1)
    batch_size = min(int(batch_size), int(X1.shape[0]))
    key = jrandom.PRNGKey(seed)
    k_flow, k_inner, k_outer, key = jrandom.split(key, 4)
    half = max(4, hidden // 2)
    flow_params = init_mlp([3] + [hidden] * num_layers + [half, 2], k_flow)
    inner_params = init_mlp([3] + [half] * num_layers + [1], k_inner)
    outer_params = init_mlp([2] + [half] * num_layers + [1], k_outer)

    flow_opt = optax.adam(lr)
    inner_opt = optax.adam(lr)
    outer_opt = optax.adam(lr)
    state = (
        flow_params,
        inner_params,
        outer_params,
        flow_opt.init(flow_params),
        inner_opt.init(inner_params),
        outer_opt.init(outer_params),
    )

    def flow_forward(params, x, t):
        return apply_mlp(params, jnp.concatenate([x, t], axis=-1))

    def inner_forward(params, x, t):
        return apply_mlp(params, jnp.concatenate([x, t], axis=-1)).squeeze(-1)

    reward_batch = jax.vmap(reward_fn_jnp)

    def continuation_grad(params, x_t, t):
        def reward_sum(x):
            x_final = integrate_flow_from(params, x, t, steps=flow_steps)
            return reward_batch(x_final).sum()

        return jax.grad(reward_sum)(x_t)

    @partial(jax.jit, static_argnames=("bc_only",))
    def step(state, key, bc_only):
        flow_p, inner_p, outer_p, flow_os, inner_os, outer_os = state
        key, k_idx, k_noise, k_t, k_pol_noise, k_pol_t = jrandom.split(key, 6)
        idx = jrandom.choice(k_idx, X1.shape[0], (batch_size,), replace=False)
        x_1 = X1[idx]
        reward = R[idx]

        def outer_loss_fn(params):
            return jnp.mean((critic_forward(params, x_1) - reward) ** 2)

        outer_loss, grads = jax.value_and_grad(outer_loss_fn)(outer_p)
        updates, outer_os = outer_opt.update(grads, outer_os, outer_p)
        outer_p = optax.apply_updates(outer_p, updates)

        x_0 = jrandom.normal(k_noise, (batch_size, 2))
        t = jrandom.uniform(k_t, (batch_size, 1))
        x_t = (1.0 - t) * x_0 + t * x_1
        x_final = integrate_flow_from(flow_p, x_t, t, steps=flow_steps)
        target_value = jax.lax.stop_gradient(critic_forward(outer_p, x_final))

        def inner_loss_fn(params):
            return jnp.mean((inner_forward(params, x_t, t) - target_value) ** 2)

        inner_loss, grads = jax.value_and_grad(inner_loss_fn)(inner_p)
        updates, inner_os = inner_opt.update(grads, inner_os, inner_p)
        inner_p = optax.apply_updates(inner_p, updates)

        z = jrandom.normal(k_pol_noise, (batch_size, 2))
        t_pol = jrandom.uniform(k_pol_t, (batch_size, 1))
        x_pol = (1.0 - t_pol) * z + t_pol * x_1
        base = x_1 - z

        def learned_sum(x):
            return inner_forward(inner_p, x, t_pol).sum()

        learned_grad = jax.grad(learned_sum)(x_pol)
        oracle_grad = continuation_grad(flow_p, x_pol, t_pol)
        chosen_grad = learned_grad if gradient_source == "learned" else oracle_grad
        weight = schedule_weight(t_pol, schedule)
        steer = weight * chosen_grad / (alpha + 1e-8)
        target = base if bc_only else base + steer

        def flow_loss_fn(params):
            pred = flow_forward(params, x_pol, t_pol)
            return jnp.mean((pred - jax.lax.stop_gradient(target)) ** 2)

        flow_loss, grads = jax.value_and_grad(flow_loss_fn)(flow_p)
        updates, flow_os = flow_opt.update(grads, flow_os, flow_p)
        flow_p = optax.apply_updates(flow_p, updates)

        learned_norm = jnp.linalg.norm(learned_grad, axis=-1)
        oracle_norm = jnp.linalg.norm(oracle_grad, axis=-1)
        grad_cos = jnp.sum(learned_grad * oracle_grad, axis=-1) / (
            learned_norm * oracle_norm + 1e-8
        )
        logs = jnp.asarray(
            [
                flow_loss,
                inner_loss,
                outer_loss,
                weight.mean(),
                jnp.linalg.norm(steer, axis=-1).mean(),
                grad_cos.mean(),
                (learned_norm / (oracle_norm + 1e-8)).mean(),
            ]
        )
        return (flow_p, inner_p, outer_p, flow_os, inner_os, outer_os), key, logs

    history = []
    for epoch in range(int(epochs)):
        state, key, logs = step(state, key, bc_only=epoch < bc_epochs)
        history.append(np.asarray(logs))

    flow_params, inner_params, outer_params, *_ = state
    z_eval = jrandom.normal(jrandom.PRNGKey(seed + 10000), (num_evals, 2))
    samples = integrate_flow(flow_params, z_eval, steps=flow_steps)
    rewards = reward_batch(samples)
    history = np.asarray(history)
    metrics = {
        "schedule": schedule,
        "gradient_source": gradient_source,
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "flow_loss": float(history[-1, 0]),
        "critic_distill_loss": float(history[-1, 1]),
        "outer_loss": float(history[-1, 2]),
        "active_weight": float(history[bc_epochs:, 3].mean()) if bc_epochs < epochs else 0.0,
        "steer_norm": float(history[bc_epochs:, 4].mean()) if bc_epochs < epochs else 0.0,
        "learned_oracle_cosine": float(history[:, 5].mean()),
        "learned_oracle_norm_ratio": float(history[:, 6].mean()),
    }
    return {
        "flow_params": flow_params,
        "inner_params": inner_params,
        "outer_params": outer_params,
        "samples": np.asarray(samples),
        "history": history,
        "metrics": metrics,
    }
