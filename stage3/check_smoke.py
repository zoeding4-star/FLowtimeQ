#!/usr/bin/env python3
"""Verify QFlow region smoke CSVs: weights, finite losses, No-Q w=0."""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stage3.guidance_weights import STAGE3_MASKS, compute_guidance_weights, describe_guidance_mask

SMOKE_GLOB = os.path.join(ROOT, "exp", "qflow-region", "qflow-region-smoke", "**", "train.csv")


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _finite(rows, keys):
    for row in rows:
        for k in keys:
            if k not in row or row[k] == "":
                continue
            try:
                v = float(row[k])
            except (TypeError, ValueError):
                return False, f"non-float {k}={row[k]!r}"
            if not math.isfinite(v):
                return False, f"non-finite {k}={v}"
    return True, ""


def main():
    paths = sorted(glob.glob(SMOKE_GLOB, recursive=True))
    if not paths:
        print("No smoke CSVs found at", SMOKE_GLOB)
        return 2

    print(f"found {len(paths)} smoke train CSVs")
    errors = []
    seen = set()
    check_keys = [
        "training/actor/total_loss",
        "training/actor/bc_loss",
        "training/actor/w_mean",
        "training/actor/guidance_weight_sum",
        "training/critic/outer_loss",
    ]
    for path in paths:
        run_dir = os.path.dirname(path)
        flags = os.path.join(run_dir, "flags.json")
        mask = None
        if os.path.exists(flags):
            with open(flags) as f:
                blob = json.load(f)
            agent = blob.get("agent") or {}
            mask = agent.get("guidance_mask_name")
        rows = _read_csv(path)
        if not rows:
            errors.append(f"empty csv {path}")
            continue
        ok, msg = _finite(rows, check_keys)
        if not ok:
            errors.append(f"{msg} in {path}")
            continue
        last = rows[-1]
        if mask is None:
            errors.append(f"missing mask in {path}")
            continue
        want = float(compute_guidance_weights(mask).sum())
        got = float(last.get("training/actor/guidance_weight_sum", "nan"))
        if abs(got - want) > 1e-2:
            errors.append(f"{mask} weight_sum {got} != {want} ({path})")
        seen.add(mask)
        if mask == "noq":
            wmean = float(last.get("training/actor/w_mean", "nan"))
            if wmean > 1e-6:
                errors.append(f"noq w_mean={wmean} should be 0 ({path})")
        dumps = glob.glob(os.path.join(run_dir, "diag_dumps", "step_*.npz"))
        if not dumps:
            errors.append(f"no diag npz in {run_dir}")
        print(
            f"ok {str(mask):14s} steps={last.get('step')} "
            f"bc={float(last.get('training/actor/bc_loss', 'nan')):.4f} "
            f"act={float(last.get('training/actor/total_loss', 'nan')):.4f} "
            f"sum_w={got:.3f}"
        )

    missing = [m for m in STAGE3_MASKS if m not in seen]
    if missing:
        errors.append(f"missing smoke masks: {missing}")
    if errors:
        print("SMOKE CHECK FAILED:")
        for e in errors:
            print(" -", e)
        return 1
    print("SMOKE CHECK PASSED", len(seen), "masks")
    for m in STAGE3_MASKS:
        print(" ", describe_guidance_mask(m)["summary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
