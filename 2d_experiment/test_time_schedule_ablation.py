#!/usr/bin/env python3
"""CPU checks for the temporal QFlow toy ablation."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from datasets import make_reward_fns, sample_from_dataset
from time_schedule_ablation import schedule_weight, train_schedule_ablation


def main():
    grid = jnp.asarray([[0.0], [0.25], [0.5], [0.75], [0.99]])
    np.testing.assert_array_equal(schedule_weight(grid, "all"), 1.0)
    np.testing.assert_array_equal(schedule_weight(grid, "noq"), 0.0)
    np.testing.assert_array_equal(schedule_weight(grid, "early50").ravel(), [1, 1, 0, 0, 0])
    np.testing.assert_array_equal(schedule_weight(grid, "late50").ravel(), [0, 0, 1, 1, 1])

    np.random.seed(0)
    data, rewards = sample_from_dataset("swissroll", 64)
    reward_fn_jnp, _, _, _ = make_reward_fns("swissroll")
    for source in ("learned", "oracle"):
        result = train_schedule_ablation(
            data,
            rewards,
            reward_fn_jnp,
            schedule="late50",
            gradient_source=source,
            epochs=3,
            bc_epochs=1,
            batch_size=16,
            hidden=8,
            num_layers=1,
            flow_steps=2,
            num_evals=16,
        )
        metrics = result["metrics"]
        assert np.isfinite(metrics["reward_mean"]), metrics
        assert result["samples"].shape == (16, 2)
        print(source, metrics)
    print("2D time-schedule ablation CPU smoke passed")


if __name__ == "__main__":
    main()
