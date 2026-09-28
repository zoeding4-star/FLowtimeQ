"""Reduce / save QFlow diagnostic snapshots."""

from __future__ import annotations

import os

import numpy as np


VECTOR_KEYS = (
    "x_interp",
    "p_interp",
    "p_flow",
    "x_flow",
    "x_term",
    "g_term",
    "dataset_actions",
)
PER_T_SAMPLE_KEYS = (
    "p_norm",
    "p_flow_norm",
    "g_outer_norm",
    "q_xt_mean",
    "q_xt_std",
    "q_outer_xt",
    "R_grad_t",
    "U_grad_t",
    "steer_norm",
    "u_norm",
    "vf_norm",
    "guide_over_bc",
    "w_t",
)
PER_SAMPLE_KEYS = (
    "q_term_mean",
    "q_term_std",
    "policy_drift",
)
SCALAR_KEYS = (
    "F_signal_early",
    "F_signal_middle",
    "F_signal_late",
    "F_signal_active",
    "F_signal_flow_early",
    "F_signal_flow_middle",
    "F_signal_flow_late",
    "guidance_weight_sum",
)

_BATCH_AXIS = {
    "x_interp": 1,
    "p_interp": 1,
    "p_norm": 1,
    "p_flow": 1,
    "p_flow_norm": 1,
    "x_flow": 1,
    "x_term": 0,
    "g_outer_norm": 1,
    "q_inner": 2,
    "q_xt_mean": 1,
    "q_xt_std": 1,
    "q_outer_xt": 1,
    "R_grad_t": 1,
    "U_grad_t": 1,
    "steer_norm": 1,
    "u_norm": 1,
    "vf_norm": 1,
    "guide_over_bc": 1,
    "w_t": 1,
    "q_term": 1,
    # Mean-ensemble action gradient is (B, A), not (M, B, A).
    "g_term": 0,
    "q_term_mean": 0,
    "q_term_std": 0,
    "policy_drift": 0,
    "dataset_actions": 0,
}


def _to_numpy(tree):
    return {k: np.asarray(v) for k, v in tree.items()}


def _moment_dict(prefix, arr):
    arr = np.asarray(arr, dtype=np.float64)
    flat = arr.reshape(-1)
    return {
        f"{prefix}_mean": float(flat.mean()),
        f"{prefix}_std": float(flat.std()),
        f"{prefix}_q10": float(np.quantile(flat, 0.10)),
        f"{prefix}_q50": float(np.quantile(flat, 0.50)),
        f"{prefix}_q90": float(np.quantile(flat, 0.90)),
    }


def reduce_snapshot(snap):
    snap = _to_numpy(snap)
    t = snap["t"]
    T = int(t.shape[0])
    row = {"flow_steps": float(T)}
    for k in range(T):
        pfx = f"t{k:02d}"
        row[f"{pfx}/t"] = float(t[k])
        for name in PER_T_SAMPLE_KEYS:
            if name in snap:
                row.update(_moment_dict(f"{pfx}/{name}", snap[name][k]))
    for name in PER_SAMPLE_KEYS:
        if name in snap:
            row.update(_moment_dict(name, snap[name]))
    for key in SCALAR_KEYS:
        if key in snap:
            row[key] = float(np.asarray(snap[key]).reshape(-1)[0])
    if "am_w" in snap:
        for k in range(T):
            row[f"t{k:02d}/w"] = float(np.asarray(snap["am_w"]).reshape(-1)[k])
    return row


def _batch_size(snap):
    x = np.asarray(snap["x_interp"])
    if x.ndim >= 2:
        return int(x.shape[1])
    return int(np.asarray(snap["x_term"]).shape[0])


def _axis_for_batch(arr, batch_size, hinted=None):
    if hinted is not None and hinted < arr.ndim and arr.shape[hinted] == batch_size:
        return hinted
    matches = [i for i, size in enumerate(arr.shape) if size == batch_size]
    if not matches:
        return None
    return matches[-1]


def subsample_snapshot(snap, n_samples, seed):
    snap = _to_numpy(snap)
    b = _batch_size(snap)
    n = min(int(n_samples), b)
    rng = np.random.RandomState(int(seed) % (2**31 - 1))
    idx = np.sort(rng.choice(b, size=n, replace=False))
    out = {"t": snap["t"].astype(np.float32), "sample_index": idx.astype(np.int32)}
    for k, v in snap.items():
        if k == "t":
            continue
        arr = np.asarray(v)
        axis = _axis_for_batch(arr, b, hinted=_BATCH_AXIS.get(k))
        if axis is None:
            out[k] = arr
            continue
        out[k] = np.take(arr, idx, axis=axis)
    for k in VECTOR_KEYS:
        if k in out:
            out[k] = np.asarray(out[k], dtype=np.float16)
    return out


def save_snapshot_npz(path, snap, step, n_samples, seed):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dumped = subsample_snapshot(snap, n_samples=n_samples, seed=seed)
    dumped["step"] = np.int64(step)
    np.savez_compressed(path, **dumped)
    return path
