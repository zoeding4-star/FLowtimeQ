#!/usr/bin/env python3
"""QFlow Stage-2 / Stage-3 job lists.

Stage 2 is a single All run with diagnostic dumps (p_t analogue).
Stage 3 is the floor+boost ablation, 1 seed per mask.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stage3.guidance_weights import STAGE3_MASKS

DEBUG_SEED = (10001,)
EXTRA_SEEDS = (20002, 30003)


def stage3_jobs(seeds=DEBUG_SEED):
    return [(mask, int(seed)) for seed in seeds for mask in STAGE3_MASKS]


def stage2_jobs(seeds=DEBUG_SEED):
    return [("all", int(seed), "diag") for seed in seeds]


def debug_jobs():
    return stage3_jobs(DEBUG_SEED)


def shard(jobs, index, n_shards):
    return [j for i, j in enumerate(jobs) if i % n_shards == index]


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("stage2", "stage3", "smoke"), default="stage3")
    p.add_argument("--shard", type=int, default=-1)
    p.add_argument("--n-shards", type=int, default=2)
    p.add_argument("--all-seeds", action="store_true")
    args = p.parse_args()
    seeds = DEBUG_SEED + EXTRA_SEEDS if args.all_seeds else DEBUG_SEED
    if args.stage == "stage2":
        jobs = stage2_jobs(seeds)
        rows = [f"{m} {s} stage2" for m, s, _kind in jobs]
    else:
        jobs = stage3_jobs(seeds)
        kind = "smoke" if args.stage == "smoke" else "stage3"
        rows = [f"{m} {s} {kind}" for m, s in jobs]
    if args.shard >= 0:
        rows = [r for i, r in enumerate(rows) if i % args.n_shards == args.shard]
    for row in rows:
        print(row)
