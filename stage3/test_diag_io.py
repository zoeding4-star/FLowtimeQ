#!/usr/bin/env python3
"""Reproduce the smoke dump IndexError without GPU/JAX."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from diag.io import subsample_snapshot


def _fake_snap(t=10, b=256, a=5, m=10):
    rng = np.random.RandomState(0)
    return {
        "t": np.linspace(0.0, 1.0, t, endpoint=False, dtype=np.float32),
        "x_interp": rng.randn(t, b, a).astype(np.float32),
        "p_interp": rng.randn(t, b, a).astype(np.float32),
        "p_norm": rng.rand(t, b).astype(np.float32),
        "p_flow": rng.randn(t, b, a).astype(np.float32),
        "p_flow_norm": rng.rand(t, b).astype(np.float32),
        "x_flow": rng.randn(t, b, a).astype(np.float32),
        "x_term": rng.randn(b, a).astype(np.float32),
        "g_outer_norm": rng.rand(t, b).astype(np.float32),
        "q_inner": rng.randn(t, m, b).astype(np.float32),
        "q_xt_mean": rng.randn(t, b).astype(np.float32),
        "q_xt_std": rng.rand(t, b).astype(np.float32),
        "q_outer_xt": rng.randn(t, b).astype(np.float32),
        "R_grad_t": rng.rand(t, b).astype(np.float32),
        "U_grad_t": rng.rand(t, b).astype(np.float32),
        "steer_norm": rng.rand(t, b).astype(np.float32),
        "u_norm": rng.rand(t, b).astype(np.float32),
        "vf_norm": rng.rand(t, b).astype(np.float32),
        "guide_over_bc": rng.rand(t, b).astype(np.float32),
        "w_t": rng.rand(t, b).astype(np.float32),
        "am_w": rng.rand(t).astype(np.float32),
        "q_term": rng.randn(m, b).astype(np.float32),
        "g_term": rng.randn(b, a).astype(np.float32),
        "q_term_mean": rng.randn(b).astype(np.float32),
        "q_term_std": rng.rand(b).astype(np.float32),
        "policy_drift": rng.rand(b).astype(np.float32),
        "dataset_actions": rng.randn(b, a).astype(np.float32),
        "F_signal_late": np.array(0.4, dtype=np.float32),
        "guidance_weight_sum": np.array(10.0, dtype=np.float32),
    }


def main():
    snap = _fake_snap()
    dumped = subsample_snapshot(snap, n_samples=16, seed=10001)
    assert dumped["g_term"].shape == (16, 5), dumped["g_term"].shape
    assert dumped["x_interp"].shape == (10, 16, 5), dumped["x_interp"].shape
    assert dumped["q_term"].shape == (10, 16), dumped["q_term"].shape
    assert dumped["q_inner"].shape == (10, 10, 16), dumped["q_inner"].shape
    assert dumped["x_term"].shape == (16, 5), dumped["x_term"].shape
    assert dumped["sample_index"].shape == (16,)
    print("subsample_snapshot ok", {k: dumped[k].shape for k in ("g_term", "q_term", "x_interp", "q_inner")})
    print("diag io test passed")


if __name__ == "__main__":
    main()
