"""Flow-time guidance weights for QFlow Stage 2/3.

Time convention (same as QAM Stage 3):

    t in [0, 1],   t_k = k / T,   k = 0, ..., T-1

t=0 is source / initial noise; t -> 1 is the terminal action. With T=10
this is {0, 0.1, ..., 0.9}.

Stage 3 does **not** use 0/1 masks (except No-Q). Every other schedule has a
strictly positive floor so the behavior-matching term is never left without a
Q-guidance residual at any t. Boosts raise one region above the floor; ramps
increase or decrease smoothly. Nonzero schedules are scaled to mean(w)=1,
matching All's total guidance budget.

Actor target (grad loss):

    v*(x_t, t) = u_t + w(t) * beta * grad_{x_t} Q_inner(s, x_t, t)

BC velocity u_t is always present. w(t) only scales the inner-Q steering term.
"""

from __future__ import annotations

import numpy as np

FLOOR_DEFAULT = 0.25

MASK_IDS = {
    "noq": "M0",
    "all": "M1",
    "early_boost": "M2",
    "middle_boost": "M3",
    "late_boost": "M4",
    "increasing": "M5",
    "decreasing": "M6",
}

ALIASES = {
    "no_q": "noq",
    "q0": "noq",
    "m0": "noq",
    "uniform": "all",
    "m1": "all",
    "early": "early_boost",
    "earlyboost": "early_boost",
    "m2": "early_boost",
    "middle": "middle_boost",
    "mid_boost": "middle_boost",
    "m3": "middle_boost",
    "late": "late_boost",
    "lateboost": "late_boost",
    "m4": "late_boost",
    "inc": "increasing",
    "late_heavy": "increasing",
    "m5": "increasing",
    "dec": "decreasing",
    "early_heavy": "decreasing",
    "m6": "decreasing",
}

# First-round Stage-3 matrix (1 seed). Stage-2 diagnostics use "all".
STAGE3_MASKS = (
    "noq",
    "all",
    "early_boost",
    "middle_boost",
    "late_boost",
    "increasing",
    "decreasing",
)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    s = str(value).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n", ""):
        return False
    raise ValueError(f"cannot parse bool from {value!r}")


def normalize_mask_name(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    return ALIASES.get(key, key)


def normalized_flow_times(flow_steps: int) -> np.ndarray:
    t = int(flow_steps)
    if t <= 0:
        raise ValueError(f"flow_steps must be positive, got {flow_steps}")
    return np.arange(t, dtype=np.float64) / float(t)


def _mean_one(w: np.ndarray) -> np.ndarray:
    m = float(w.mean())
    if m <= 0.0:
        return w
    return w / m


def compute_guidance_weights(
    name: str,
    flow_steps: int = 10,
    floor: float = FLOOR_DEFAULT,
) -> np.ndarray:
    """Return w with shape (T,). No-Q is all zeros; every other mask is > 0 and mean 1."""
    name = normalize_mask_name(name)
    T = int(flow_steps)
    t = normalized_flow_times(T)
    floor = float(floor)
    if not (0.0 < floor < 1.0):
        raise ValueError(f"guidance floor must be in (0, 1), got {floor}")

    if name == "noq":
        return np.zeros(T, dtype=np.float64)
    if name == "all":
        return np.ones(T, dtype=np.float64)

    highlight = np.zeros(T, dtype=np.float64)
    if name == "early_boost":
        highlight = (t < 0.5).astype(np.float64)
        w = floor + (1.0 - floor) * highlight
    elif name == "middle_boost":
        highlight = ((t >= 0.3) & (t < 0.7)).astype(np.float64)
        w = floor + (1.0 - floor) * highlight
    elif name == "late_boost":
        highlight = (t >= 0.5).astype(np.float64)
        w = floor + (1.0 - floor) * highlight
    elif name == "increasing":
        w = floor + (1.0 - floor) * t
    elif name == "decreasing":
        w = floor + (1.0 - floor) * (1.0 - t)
    else:
        raise ValueError(
            f"Unknown guidance_mask_name={name!r}. Known: {sorted(MASK_IDS)}"
        )
    return _mean_one(w)


def region_masks(flow_steps: int = 10):
    t = normalized_flow_times(flow_steps)
    return {
        "early": t < 0.5,
        "middle": (t >= 0.3) & (t < 0.7),
        "late": t >= 0.5,
    }


def f_signal(p_norm_t: np.ndarray, region: np.ndarray, eps: float = 1e-8) -> float:
    """F_signal(S) = sum_{t in S} ||p_t|| / sum_t ||p_t||."""
    p = np.asarray(p_norm_t, dtype=np.float64).reshape(-1)
    m = np.asarray(region, dtype=np.float64).reshape(-1)
    denom = float(p.sum()) + eps
    return float((p * m).sum() / denom)


def describe_guidance_mask(
    name: str,
    flow_steps: int = 10,
    floor: float = FLOOR_DEFAULT,
) -> dict:
    w = compute_guidance_weights(name, flow_steps=flow_steps, floor=floor)
    t = normalized_flow_times(flow_steps)
    name = normalize_mask_name(name)
    active = w > 0
    n_active = int(active.sum())
    if n_active > 0:
        t_lo = float(t[active].min())
        t_hi = float(t[active].max())
        active_steps = np.nonzero(active)[0].tolist()
    else:
        t_lo, t_hi, active_steps = 0.0, 0.0, []
    weights = tuple(float(x) for x in w)
    mid = MASK_IDS.get(name, name)
    summary = (
        f"{mid} {name} T={flow_steps} n_active={n_active} "
        f"t_active=[{t_lo:.2f},{t_hi:.2f}] min(w)={float(w.min()):.4f} "
        f"sum(w)={float(w.sum()):.4f} floor={floor} w={weights}"
    )
    return {
        "name": name,
        "id": mid,
        "flow_steps": int(flow_steps),
        "floor": float(floor),
        "t": tuple(float(x) for x in t),
        "weights": weights,
        "n_active": n_active,
        "active_steps": active_steps,
        "t_lo": t_lo,
        "t_hi": t_hi,
        "weight_sum": float(w.sum()),
        "weight_min": float(w.min()),
        "summary": summary,
    }
