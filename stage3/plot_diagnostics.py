#!/usr/bin/env python3
"""Plot QFlow Stage-2 p_t analogue: interpolant and Euler ||nabla Q_inner||."""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_dumps(dump_dir):
    paths = sorted(glob.glob(os.path.join(dump_dir, "step_*.npz")))
    dumps = []
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            dumps.append({k: z[k] for k in z.files})
    return dumps


def _mean_quantile(ax, t, values, label, color="C0"):
    mean = values.mean(axis=1)
    q10 = np.quantile(values, 0.10, axis=1)
    q90 = np.quantile(values, 0.90, axis=1)
    ax.fill_between(t, q10, q90, color=color, alpha=0.22, label="q10–q90")
    ax.plot(t, mean, color=color, lw=2.2, label=label)
    ax.plot(t, np.quantile(values, 0.50, axis=1), color=color, lw=1.2, ls="--", alpha=0.8, label="median")


def _panel(ax, t, values, title, ylabel, rng):
    T, n = values.shape
    tt = np.repeat(np.asarray(t).reshape(-1, 1), n, axis=1)
    jitter = (rng.rand(T, n) - 0.5) * (0.6 * (t[1] - t[0] if T > 1 else 0.04))
    ax.scatter((tt + jitter).ravel(), values.ravel(), s=6, alpha=0.08, c="C0", linewidths=0)
    _mean_quantile(ax, t, values, "mean")
    ax.set_title(title)
    ax.set_xlabel("flow time t")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump-dir", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    dumps = _load_dumps(args.dump_dir)
    if not dumps:
        raise SystemExit(f"no dumps in {args.dump_dir}")
    out = args.out or os.path.join(os.path.dirname(args.dump_dir), "figures")
    os.makedirs(out, exist_ok=True)
    latest = dumps[-1]
    t = latest["t"].astype(np.float64)
    rng = np.random.RandomState(0)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    _panel(
        axes[0],
        t,
        np.asarray(latest["p_norm"], dtype=np.float64),
        r"A  interpolant $\|\nabla_{x_t} Q_i\|$  (train $p_t$)",
        r"$\|p_t\|$",
        rng,
    )
    _panel(
        axes[1],
        t,
        np.asarray(latest["p_flow_norm"], dtype=np.float64),
        r"B  Euler $\|\nabla_{x_t} Q_i\|$  (eval path)",
        r"$\|p_t^{\mathrm{flow}}\|$",
        rng,
    )
    _panel(
        axes[2],
        t,
        np.asarray(latest["R_grad_t"], dtype=np.float64),
        r"C  inner-Q $R_{\mathrm{grad}}(t)$",
        r"$R_{\mathrm{grad}}$",
        rng,
    )
    fig.suptitle(f"QFlow Stage 2  (latest dump step={int(latest['step'])})", y=1.03)
    fig.tight_layout()
    png = os.path.join(out, "fig_qflow_pt.png")
    fig.savefig(png, dpi=160)
    fig.savefig(os.path.splitext(png)[0] + ".pdf")
    plt.close(fig)
    print("wrote", png)

    # F_signal table from latest mean ||p||
    p = np.asarray(latest["p_norm"], dtype=np.float64).mean(axis=1)
    pf = np.asarray(latest["p_flow_norm"], dtype=np.float64).mean(axis=1)
    csv_path = os.path.join(out, "latest_t_moments.csv")
    with open(csv_path, "w") as f:
        f.write("t,p_interp_mean,p_flow_mean,R_grad_mean,guide_over_bc_mean\n")
        rg = np.asarray(latest["R_grad_t"], dtype=np.float64).mean(axis=1)
        gob = np.asarray(latest["guide_over_bc"], dtype=np.float64).mean(axis=1)
        for k in range(len(t)):
            f.write(f"{t[k]:.3f},{p[k]:.6f},{pf[k]:.6f},{rg[k]:.6f},{gob[k]:.6f}\n")
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
