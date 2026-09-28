"""QFlow with flow-time guidance weights and p_t diagnostics.

Does not modify `agents/qflow.py`. BC velocity is always the actor target;
`w(t)` only scales the inner-critic steering term:

    v* = u_t + w(t) * beta * nabla_{x_t} Q_inner(s, x_t, t)

QFlow's p_t analogue is that inner-Q gradient (training interpolant and
Euler policy trajectory).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from agents.qflow import QFlowAgent
from agents.qflow import get_config as qflow_get_config
from stage3.guidance_weights import describe_guidance_mask


def _unit_dir_agreement(g, eps=1e-8):
    """g: (M, B, A) -> R_grad (B,), U_grad (B,)."""
    g_norm = jnp.linalg.norm(g, axis=-1, keepdims=True)
    ghat = g / (g_norm + eps)
    mean_dir = jnp.mean(ghat, axis=0)
    r_grad = jnp.linalg.norm(mean_dir, axis=-1)
    return r_grad, 1.0 - r_grad


class QFlowRegionAgent(QFlowAgent):
    """QFlow whose inner-Q guidance is weighted in normalized flow time."""

    def _guidance_weight(self, t):
        w = jnp.asarray(self.config["guidance_weights"], dtype=jnp.float32)
        t = jnp.asarray(t)
        if t.ndim == 1:
            t = t[:, None]
        t_idx = jnp.clip(
            jnp.floor(t * w.shape[0]).astype(jnp.int32),
            0,
            w.shape[0] - 1,
        )
        return w[t_idx.reshape(-1)].reshape((-1, 1))

    def _inner_q_and_grad(self, observations, x_t, t):
        t_features = self._get_time_features(t)
        q = self.network.select("tc_critic")(observations, x_t, t_features)
        grad = jax.grad(
            lambda actions: self.network.select("tc_critic")(
                observations, actions, t_features
            )
            .mean(axis=0)
            .sum()
        )(x_t)
        return q, grad

    def _outer_q_and_grad(self, observations, actions):
        a = jnp.clip(actions, -1.0, 1.0)
        q = self.network.select("critic")(observations, a)
        grad = jax.grad(
            lambda act: self.network.select("critic")(observations, act).mean(axis=0).sum()
        )(a)
        return q, grad

    def actor_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch["actions"].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch["actions"]
        t = jax.random.uniform(t_rng, (batch_size, 1), minval=0.0, maxval=1.0)
        x_t = (1.0 - t) * x_0 + t * x_1
        u_t = x_1 - x_0
        observations = batch["observations"]

        pred_vel = self.network.select("actor_bc_flow")(
            observations, x_t, t, is_encoded=True, params=grad_params
        )
        loss_type = self.config["actor_loss_type"]
        if loss_type != "grad":
            raise ValueError(
                f"qflow_region only implements actor_loss_type='grad', got {loss_type}"
            )

        t_features = self._get_time_features(t)
        q_grad = jax.grad(
            lambda actions: self.network.select("tc_critic")(
                observations, actions, t_features
            )
            .mean(axis=0)
            .sum()
        )
        grad_v = jax.lax.stop_gradient(q_grad(x_t))
        w_t = self._guidance_weight(t)
        beta = 1.0 / (self.config["guidance_lambda"] + 1e-8)
        steer = w_t * beta * grad_v
        v_target = u_t + steer
        total_loss = jnp.mean((pred_vel - jax.lax.stop_gradient(v_target)) ** 2)
        bc_loss = jnp.mean((pred_vel - u_t) ** 2)
        p_norm = jnp.linalg.norm(grad_v, axis=-1)
        steer_norm = jnp.linalg.norm(steer, axis=-1)
        u_norm = jnp.linalg.norm(u_t, axis=-1)
        info = {
            "total_loss": total_loss,
            "bc_loss": bc_loss,
            "w_mean": jnp.mean(w_t),
            "guidance_weight_sum": jnp.sum(jnp.asarray(self.config["guidance_weights"])),
            "p_norm_mean": jnp.mean(p_norm),
            "steer_norm_mean": jnp.mean(steer_norm),
            "u_norm_mean": jnp.mean(u_norm),
            "guide_over_bc": jnp.mean(steer_norm / (u_norm + 1e-8)),
            "inner_q_at_xt": self.network.select("tc_critic")(
                observations, x_t, t_features
            )
            .mean(axis=0)
            .mean(),
        }
        return total_loss, info

    @jax.jit
    def diagnostic_snapshot(self, batch, rng):
        """Raw flow-time signals. Not used by the training gradient."""
        observations = batch["observations"]
        x_1 = batch["actions"]
        batch_size, action_dim = x_1.shape
        T = int(self.config["flow_steps"])
        eps = 1e-8
        rng, n0, nflow = jax.random.split(rng, 3)
        x_0 = jax.random.normal(n0, (batch_size, action_dim))
        t_grid = jnp.arange(T) / float(T)
        w = jnp.asarray(self.config["guidance_weights"], dtype=jnp.float32)
        beta = 1.0 / (self.config["guidance_lambda"] + 1e-8)

        def one_t(tk):
            t = jnp.full((batch_size, 1), tk)
            xt = (1.0 - t) * x_0 + t * x_1
            q_in, g_in = self._inner_q_and_grad(observations, xt, t)
            q_out, g_out = self._outer_q_and_grad(observations, xt)
            vel = self.network.select("actor_bc_flow")(
                observations, xt, t, is_encoded=True
            )
            u = x_1 - x_0
            wt = self._guidance_weight(t)
            p_norm = jnp.linalg.norm(g_in, axis=-1)
            g_out_norm = jnp.linalg.norm(g_out, axis=-1)
            steer_norm = jnp.linalg.norm(wt * beta * g_in, axis=-1)
            u_norm = jnp.linalg.norm(u, axis=-1)
            vf_norm = jnp.linalg.norm(vel, axis=-1)
            q_inner_mean = q_in.mean(axis=0)
            q_inner_std = q_in.std(axis=0)
            q_outer_mean = q_out.mean(axis=0)
            return {
                "xt": xt,
                "p": g_in,
                "p_norm": p_norm,
                "g_outer_norm": g_out_norm,
                "q_inner": q_inner_mean,
                "q_inner_std": q_inner_std,
                "q_outer": q_outer_mean,
                "steer_norm": steer_norm,
                "u_norm": u_norm,
                "vf_norm": vf_norm,
                "guide_over_bc": steer_norm / (u_norm + eps),
                "w": wt.reshape(-1),
            }

        # Per-ensemble inner gradient agreement on interpolant.
        def inner_ensemble_grad(tk):
            t = jnp.full((batch_size, 1), tk)
            xt = (1.0 - t) * x_0 + t * x_1
            t_features = self._get_time_features(t)
            q = self.network.select("tc_critic")(observations, xt, t_features)
            n_ens = int(self.config["num_ensembles"])

            def grad_from_weights(e):
                wgt = jnp.zeros((n_ens,), dtype=xt.dtype).at[e].set(1.0)

                def scalar(act):
                    qq = self.network.select("tc_critic")(
                        observations, act, t_features
                    )
                    return (qq * wgt[:, None]).sum()

                return jax.grad(scalar)(xt)

            g = jax.vmap(grad_from_weights)(jnp.arange(n_ens))
            r, u = _unit_dir_agreement(g)
            return r, u, q

        interp = jax.vmap(one_t)(t_grid)
        r_grad_t, u_grad_t, q_ens_t = jax.vmap(inner_ensemble_grad)(t_grid)

        # Euler policy trajectory (eval path).
        x = jax.random.normal(nflow, (batch_size, action_dim))
        xs_flow = []
        p_flow = []
        p_flow_norm = []
        for i in range(T):
            t = jnp.full((batch_size, 1), i / float(T))
            _, g_in = self._inner_q_and_grad(observations, x, t)
            xs_flow.append(x)
            p_flow.append(g_in)
            p_flow_norm.append(jnp.linalg.norm(g_in, axis=-1))
            vels = self.network.select("actor_bc_flow")(
                observations, x, t, is_encoded=True
            )
            x = x + vels / float(T)
        x_term = jnp.clip(x, -1.0, 1.0)
        xs_flow = jnp.stack(xs_flow, axis=0)
        p_flow = jnp.stack(p_flow, axis=0)
        p_flow_norm = jnp.stack(p_flow_norm, axis=0)

        q_term, g_term = self._outer_q_and_grad(observations, x_term)
        p_interp = interp["p"]
        p_interp_norm = interp["p_norm"]
        p_mean = jnp.mean(p_interp_norm, axis=1)
        p_flow_mean = jnp.mean(p_flow_norm, axis=1)
        masks = {
            "early": t_grid < 0.5,
            "middle": (t_grid >= 0.3) & (t_grid < 0.7),
            "late": t_grid >= 0.5,
            "active": w > 0,
        }

        def fsig(vals, mask):
            return jnp.sum(vals * mask.astype(vals.dtype)) / (jnp.sum(vals) + eps)

        policy_drift = jnp.linalg.norm(x_term - jnp.clip(x_1, -1.0, 1.0), axis=-1)

        return {
            "t": t_grid,
            "x_interp": interp["xt"],
            "p_interp": p_interp,
            "p_norm": p_interp_norm,  # Stage-2 analogue of QAM ||p_t|| (train interpolant)
            "p_flow": p_flow,
            "p_flow_norm": p_flow_norm,
            "x_flow": xs_flow,
            "x_term": x_term,
            "g_outer_norm": interp["g_outer_norm"],
            "q_inner": q_ens_t,
            "q_xt_mean": interp["q_inner"],
            "q_xt_std": interp["q_inner_std"],
            "q_outer_xt": interp["q_outer"],
            "R_grad_t": r_grad_t,
            "U_grad_t": u_grad_t,
            "steer_norm": interp["steer_norm"],
            "u_norm": interp["u_norm"],
            "vf_norm": interp["vf_norm"],
            "guide_over_bc": interp["guide_over_bc"],
            "w_t": interp["w"],
            "am_w": w,
            "q_term": q_term,
            "g_term": g_term,
            "q_term_mean": q_term.mean(axis=0),
            "q_term_std": q_term.std(axis=0),
            "policy_drift": policy_drift,
            "F_signal_early": fsig(p_mean, masks["early"]),
            "F_signal_middle": fsig(p_mean, masks["middle"]),
            "F_signal_late": fsig(p_mean, masks["late"]),
            "F_signal_active": fsig(p_mean, masks["active"]),
            "F_signal_flow_early": fsig(p_flow_mean, masks["early"]),
            "F_signal_flow_middle": fsig(p_flow_mean, masks["middle"]),
            "F_signal_flow_late": fsig(p_flow_mean, masks["late"]),
            "guidance_weight_sum": jnp.sum(w),
            "dataset_actions": x_1,
        }

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        T = int(config["flow_steps"])
        spec = describe_guidance_mask(
            str(config.get("guidance_mask_name", "all")),
            flow_steps=T,
            floor=float(config.get("guidance_floor", 0.25)),
        )
        config["guidance_mask_name"] = spec["name"]
        config["guidance_weights"] = spec["weights"]
        config["guidance_n_active"] = spec["n_active"]
        config["guidance_weight_sum"] = spec["weight_sum"]
        config["guidance_weight_min"] = spec["weight_min"]
        config["guidance_t_lo"] = spec["t_lo"]
        config["guidance_t_hi"] = spec["t_hi"]
        print("[qflow-region]", spec["summary"], flush=True)
        try:
            import wandb

            if wandb.run is not None:
                wandb.config.update(
                    {
                        "guidance_mask_id": spec["id"],
                        "guidance_mask_name": spec["name"],
                        "guidance_weights": list(spec["weights"]),
                        "guidance_n_active": spec["n_active"],
                        "guidance_weight_sum": spec["weight_sum"],
                        "guidance_weight_min": spec["weight_min"],
                    },
                    allow_val_change=True,
                )
        except Exception as exc:
            print("[qflow-region] wandb.config.update skipped:", exc, flush=True)
        return super().create(seed, ex_observations, ex_actions, config)


def get_config():
    config = qflow_get_config()
    config.agent_name = "qflow_region"
    config.guidance_mask_name = "all"
    config.guidance_floor = 0.25
    config.guidance_weights = (1.0,) * 10
    config.guidance_n_active = 10
    config.guidance_weight_sum = 10.0
    config.guidance_weight_min = 1.0
    config.guidance_t_lo = 0.0
    config.guidance_t_hi = 0.9
    return config
