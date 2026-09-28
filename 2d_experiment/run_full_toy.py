#!/usr/bin/env python3
"""Full 2D toy reproduction: QFlow vs FQL vs FBRAC on all four datasets.

Uses the notebook CONFIGURATION hyperparameters. Training cells in the notebook
override ALPHAS to [1, 0.5]; this script uses the CONFIGURATION list
[5, 1, 0.5, 0.3] so the complete alpha sweep is included.
"""
from __future__ import annotations

import os
import pickle
import traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.random as jrandom

from networks import DOMAIN_MIN, DOMAIN_MAX, onestep_forward, integrate_flow
from datasets import sample_from_dataset, make_reward_fns
from agents import train_fql, train_fbrac, train_qflow

# --- CONFIGURATION (matches toy_experiment_notebook.ipynb) ---
FLOW_STEPS = 25
NUM_LAYERS = 4
HIDDEN_DIM = 512
BC_EPOCHS = 2000
OFFLINE_EPOCHS = 100
LEARNING_RATE = 3e-4
BATCH_SIZE = 4096
NUM_PLOTS = 10
NUM_EVALS = 512
SAVE_DIR = "toy_experiment"
ALPHAS = [5, 1, 0.5, 0.3]
USE_TARGET_NETWORK = False
TAU = 0.005
N_DATA = 10000
TASKS = ["moons", "two_spirals", "swissroll", "eight_gaussians"]

EPOCHS = BC_EPOCHS + OFFLINE_EPOCHS
PLOT_STEP = EPOCHS // NUM_PLOTS if NUM_PLOTS > 0 else None
QFLOW_PLOT_STEP = EPOCHS // 2  # notebook sets NUM_PLOTS = 2 for QFlow


def save_dataset_plot(dataset_type: str) -> None:
    np.random.seed(42)
    samples, rewards = sample_from_dataset(dataset_type, 2000)
    rewards = rewards.flatten()
    plt.figure(figsize=(4, 4))
    plt.scatter(
        samples[:, 0], samples[:, 1],
        c=rewards, cmap="plasma", s=10, alpha=0.6, edgecolors="none",
    )
    plt.title(f"Dataset Samples\nMean reward: {rewards.mean():.3f}")
    plt.xlabel("Action dim 1")
    plt.ylabel("Action dim 2")
    plt.xlim(DOMAIN_MIN, DOMAIN_MAX)
    plt.ylim(DOMAIN_MIN, DOMAIN_MAX)
    plt.tight_layout()
    out_dir = os.path.join(SAVE_DIR, dataset_type)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "training_dataset.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}", flush=True)


def save_comparison(method: str, dataset_type: str, results, sample_fn) -> None:
    """BC | alpha=... scatter grid, matching the notebook comparison cells."""
    results_sorted = sorted(results, key=lambda x: x[0], reverse=True)
    n_cols = 1 + len(results_sorted)
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3))
    if n_cols == 1:
        axes = [axes]
    z_eval = jrandom.normal(jrandom.PRNGKey(999), (NUM_EVALS, 2))

    bc_params = results_sorted[0][2]
    bc_actions = np.array(sample_fn(bc_params, z_eval))
    axes[0].scatter(bc_actions[:, 0], bc_actions[:, 1], s=10, alpha=0.6, edgecolors="none")
    axes[0].set_title("BC", fontsize=15, fontstyle="italic")
    axes[0].set_xlim(DOMAIN_MIN, DOMAIN_MAX)
    axes[0].set_ylim(DOMAIN_MIN, DOMAIN_MAX)
    axes[0].set_aspect("equal")

    for i, (alpha, params, _) in enumerate(results_sorted):
        actions = np.array(sample_fn(params, z_eval))
        ax = axes[i + 1]
        ax.scatter(actions[:, 0], actions[:, 1], s=10, alpha=0.6, edgecolors="none")
        ax.set_title(f"α={alpha}", fontsize=15, fontstyle="italic")
        ax.set_xlim(DOMAIN_MIN, DOMAIN_MAX)
        ax.set_ylim(DOMAIN_MIN, DOMAIN_MAX)
        ax.set_aspect("equal")

    plt.tight_layout()
    out_dir = os.path.join(SAVE_DIR, dataset_type, method)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "comparison.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}", flush=True)


def dump_weights(method: str, dataset_type: str, alpha, payload: dict) -> None:
    out_dir = os.path.join(SAVE_DIR, dataset_type, method)
    os.makedirs(out_dir, exist_ok=True)
    path = (
        f"{out_dir}/weights_alpha{alpha}_flow{FLOW_STEPS}_layers{NUM_LAYERS}"
        f"_hidden{HIDDEN_DIM}_data{N_DATA}_epochs{EPOCHS}.pkl"
    )
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"  saved {path}", flush=True)


def flow_sample(params, z):
    if isinstance(params, tuple):
        params = params[0]
    return integrate_flow(params, z, steps=FLOW_STEPS)


def onestep_sample(params, z):
    return onestep_forward(params, z)


def run_task(dataset_type: str) -> None:
    print(f"\n========== TASK: {dataset_type} ==========", flush=True)
    make_reward_fns(dataset_type)
    save_dataset_plot(dataset_type)

    np.random.seed(0)
    x_1_data, rewards_data = sample_from_dataset(dataset_type, N_DATA)
    rewards_data = np.array(rewards_data).flatten()
    print(
        f"  generated {len(x_1_data)} samples "
        f"(mean reward: {rewards_data.mean():.3f})",
        flush=True,
    )

    common = dict(
        x_1_data=x_1_data,
        rewards_data=rewards_data,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        hidden=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        flow_steps=FLOW_STEPS,
        use_target_network=USE_TARGET_NETWORK,
        tau=TAU,
        batch_size=BATCH_SIZE,
        bc_epochs=BC_EPOCHS,
        num_evals=NUM_EVALS,
        save_dir=SAVE_DIR,
        dataset_type=dataset_type,
    )

    fql_results = []
    for alpha in ALPHAS:
        print(f"\n--- FQL  {dataset_type}  alpha={alpha} ---", flush=True)
        flow_p, policy_p, critic_p, bc_p, _ = train_fql(
            **common, alpha=alpha, plot_every=PLOT_STEP,
        )
        dump_weights("fql", dataset_type, alpha, {
            "flow_params": flow_p,
            "policy_params": policy_p,
            "critic_params": critic_p,
            "bc_policy_params": bc_p,
        })
        fql_results.append((alpha, policy_p, bc_p))
    save_comparison("fql", dataset_type, fql_results, onestep_sample)

    fbrac_results = []
    for alpha in ALPHAS:
        print(f"\n--- FBRAC  {dataset_type}  alpha={alpha} ---", flush=True)
        flow_p, critic_p, bc_p, _ = train_fbrac(
            **common, alpha=alpha, plot_every=PLOT_STEP,
        )
        dump_weights("fbrac", dataset_type, alpha, {
            "flow_params": flow_p,
            "critic_params": critic_p,
            "bc_policy_params": bc_p,
        })
        fbrac_results.append((alpha, flow_p, bc_p))
    save_comparison("fbrac", dataset_type, fbrac_results, flow_sample)

    qflow_results = []
    for alpha in ALPHAS:
        print(f"\n--- QFlow  {dataset_type}  alpha={alpha} ---", flush=True)
        flow_p, critic_p, bc_p, _ = train_qflow(
            **common, alpha=alpha, plot_every=QFLOW_PLOT_STEP,
            use_time_embed=False, time_embed_dim=16,
        )
        dump_weights("qflow", dataset_type, alpha, {
            "flow_params": flow_p,
            "inner_critic_params": critic_p,
            "bc_policy_params": bc_p,
        })
        qflow_results.append((alpha, flow_p, bc_p))
    save_comparison("qflow", dataset_type, qflow_results, flow_sample)


def main() -> None:
    print("JAX devices:", jax.devices(), flush=True)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
    print("Tasks:", TASKS, flush=True)
    print("Alphas:", ALPHAS, flush=True)
    print(f"Epochs: {EPOCHS} (BC {BC_EPOCHS} + offline {OFFLINE_EPOCHS})", flush=True)

    failed = []
    for task in TASKS:
        try:
            run_task(task)
        except Exception:
            traceback.print_exc()
            failed.append(task)

    print("\n========== DONE ==========", flush=True)
    print("Output dir:", os.path.abspath(SAVE_DIR), flush=True)
    if failed:
        print("FAILED tasks:", failed, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
