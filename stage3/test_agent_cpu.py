#!/usr/bin/env python3
"""CPU smoke of QFlowRegionAgent.create / actor_loss / snapshot. No GPU."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from agents.qflow_region import QFlowRegionAgent, get_config
from diag.io import reduce_snapshot, subsample_snapshot


def _tiny_config(mask):
    config = get_config()
    config.actor_hidden_dims = (16, 16)
    config.value_hidden_dims = (16, 16)
    config.flow_steps = 4
    config.guidance_mask_name = mask
    config.guidance_floor = 0.25
    config.use_time_embed = True
    config.time_embed_dim = 16
    config.num_ensembles = 2
    return config


def main():
    rng = jax.random.PRNGKey(0)
    obs = jnp.zeros((8, 6), dtype=jnp.float32)
    act = jnp.zeros((8, 3), dtype=jnp.float32)
    batch = {"observations": obs, "actions": act}

    for mask in ("noq", "all", "early_boost"):
        agent = QFlowRegionAgent.create(0, obs[:1], act[:1], _tiny_config(mask))
        w = np.asarray(agent.config["guidance_weights"])
        if mask == "noq":
            assert np.allclose(w, 0)
        else:
            assert np.all(w > 0)
        loss, info = agent.actor_loss(batch, agent.network.params, rng)
        assert np.isfinite(float(loss))
        assert np.isfinite(float(info["p_norm_mean"]))
        if mask == "noq":
            assert float(info["w_mean"]) == 0.0
        snap = agent.diagnostic_snapshot(batch, rng)
        assert snap["p_norm"].shape[0] == 4
        assert snap["p_flow_norm"].shape[0] == 4
        snap_np = jax.device_get(snap)
        reduce_snapshot(snap_np)
        dumped = subsample_snapshot(snap_np, n_samples=4, seed=0)
        assert dumped["g_term"].shape[0] == 4
        assert dumped["g_term"].shape[-1] == 3
        print(f"ok {mask:12s} loss={float(loss):.4f} w_mean={float(info['w_mean']):.3f} "
              f"p={float(info['p_norm_mean']):.4f} F_late={float(snap['F_signal_late']):.3f}")
    print("agent cpu smoke passed")


if __name__ == "__main__":
    main()
