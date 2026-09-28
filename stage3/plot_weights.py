#!/usr/bin/env python3
"""Plot QFlow Stage-3 guidance weights (never-zero except No-Q)."""

from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stage3.guidance_weights import STAGE3_MASKS, compute_guidance_weights, normalized_flow_times

OUT = os.path.join(ROOT, "notes", "figures", "qflow_stage3")


def main():
    os.makedirs(OUT, exist_ok=True)
    t = normalized_flow_times(10)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for name in STAGE3_MASKS:
        w = compute_guidance_weights(name)
        ax.plot(t, w, marker="o", lw=2, label=name)
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.4)
    ax.set_xlabel("flow time t")
    ax.set_ylabel("guidance weight w(t)")
    ax.set_title("QFlow Stage 3: floor+boost / ramps (No-Q is the only zero schedule)")
    ax.grid(True, alpha=0.25)
    ax.legend(ncols=2, fontsize=8)
    fig.tight_layout()
    png = os.path.join(OUT, "fig_guidance_weights.png")
    fig.savefig(png, dpi=160)
    fig.savefig(os.path.splitext(png)[0] + ".pdf")
    plt.close(fig)
    print("wrote", png)


if __name__ == "__main__":
    main()
