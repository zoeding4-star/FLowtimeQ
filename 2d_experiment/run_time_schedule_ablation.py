#!/usr/bin/env python3
"""Run the 6 schedules x 2 gradient-source 2D QFlow screen on CPU."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from datasets import make_reward_fns, sample_from_dataset
from time_schedule_ablation import GRADIENT_SOURCES, SCHEDULES, train_schedule_ablation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="swissroll", choices=("swissroll", "two_spirals"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--bc-epochs", type=int, default=40)
    parser.add_argument("--n-data", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/time_schedule")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    x_1, rewards = sample_from_dataset(args.dataset, args.n_data)
    reward_fn_jnp, _, _, _ = make_reward_fns(args.dataset)
    output = Path(args.output) / args.dataset
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    for source in GRADIENT_SOURCES:
        for schedule in SCHEDULES:
            print(f"running source={source} schedule={schedule}", flush=True)
            result = train_schedule_ablation(
                x_1,
                rewards,
                reward_fn_jnp,
                schedule=schedule,
                gradient_source=source,
                epochs=args.epochs,
                bc_epochs=args.bc_epochs,
                batch_size=args.batch_size,
                hidden=args.hidden,
                flow_steps=args.flow_steps,
                alpha=args.alpha,
                seed=args.seed,
            )
            rows.append(result["metrics"])
            np.save(output / f"samples_{source}_{schedule}.npy", result["samples"])

    with (output / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    x = np.arange(len(SCHEDULES))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 4))
    for idx, source in enumerate(GRADIENT_SOURCES):
        baseline = next(
            r["reward_mean"] for r in rows
            if r["gradient_source"] == source and r["schedule"] == "noq"
        )
        values = [
            1000.0 * (
                next(r["reward_mean"] for r in rows if r["gradient_source"] == source and r["schedule"] == s)
                - baseline
            )
            for s in SCHEDULES
        ]
        ax.bar(x + (idx - 0.5) * width, values, width, label=source)
    ax.set_xticks(x, SCHEDULES, rotation=20)
    ax.set_ylabel("terminal reward improvement over NoQ (×1e-3)")
    ax.set_title(f"QFlow time schedule ablation: {args.dataset}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "reward_comparison.png", dpi=150)
    plt.close(fig)
    print(f"saved results to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
