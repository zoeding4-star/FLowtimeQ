#!/usr/bin/env python3
"""CPU tests for QFlow guidance schedules. No GPU."""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stage3.guidance_weights import (
    STAGE3_MASKS,
    compute_guidance_weights,
    describe_guidance_mask,
    f_signal,
    region_masks,
)


def _assert_close(a, b, tol=1e-8):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or np.max(np.abs(a - b)) > tol:
        raise AssertionError(f"not close:\n{a}\n{b}")


def test_noq_and_all():
    _assert_close(compute_guidance_weights("noq"), np.zeros(10))
    _assert_close(compute_guidance_weights("all"), np.ones(10))


def test_nonzero_except_noq():
    for name in STAGE3_MASKS:
        w = compute_guidance_weights(name)
        assert w.shape == (10,), name
        if name == "noq":
            assert np.all(w == 0)
        else:
            assert np.all(w > 0), (name, w)
            assert abs(w.mean() - 1.0) < 1e-12, (name, w.mean())


def test_boosts_raise_region():
    t = np.arange(10) / 10.0
    early = compute_guidance_weights("early_boost")
    late = compute_guidance_weights("late_boost")
    mid = compute_guidance_weights("middle_boost")
    assert early[t < 0.5].mean() > early[t >= 0.5].mean()
    assert late[t >= 0.5].mean() > late[t < 0.5].mean()
    m = (t >= 0.3) & (t < 0.7)
    assert mid[m].mean() > mid[~m].mean()


def test_ramps():
    inc = compute_guidance_weights("increasing")
    dec = compute_guidance_weights("decreasing")
    assert np.all(np.diff(inc) > 0)
    assert np.all(np.diff(dec) < 0)
    assert inc[0] > 0.05 and dec[-1] > 0.05


def test_aliases_and_describe():
    _assert_close(compute_guidance_weights("q0"), compute_guidance_weights("noq"))
    _assert_close(compute_guidance_weights("early-heavy"), compute_guidance_weights("decreasing"))
    d = describe_guidance_mask("late_boost")
    assert d["n_active"] == 10
    assert d["weight_min"] > 0


def test_f_signal():
    p = np.array([0.01, 0.05, 0.15, 0.33, 0.69, 1.14, 1.82, 3.23, 5.64, 7.77])
    m = region_masks(10)
    fe = f_signal(p, m["early"])
    fl = f_signal(p, m["late"])
    assert 0.04 < fe < 0.08, fe
    assert 0.90 < fl < 0.96, fl


if __name__ == "__main__":
    test_noq_and_all()
    test_nonzero_except_noq()
    test_boosts_raise_region()
    test_ramps()
    test_aliases_and_describe()
    test_f_signal()
    print("guidance_weights tests passed")
    for name in STAGE3_MASKS:
        print(" ", describe_guidance_mask(name)["summary"])
