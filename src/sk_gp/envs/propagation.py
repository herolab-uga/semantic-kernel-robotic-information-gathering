"""Semantics-aware Wi-Fi RSSI propagation over an :class:`Environment`.

Implements a log-distance path-loss model with material-dependent attenuation
(the deterministic core of Eq. 12).  A ray from the access point to a query
location accumulates a per-crossing attenuation for every wall/object it
intersects::

    RSSI(x) = P_tx - 10 * zeta * log10(d(x, x_AP))  -  sum_{b crossed} A(material_b)

Multi-AP fields are fused by max-selection (best AP) or by power-summation.  The
stochastic large-/small-scale fading terms of Eq. 12 are added by the simulator's
:func:`~sk_gp.simulator.radio_world.add_fading_noise` when synthesizing training
measurements.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .environment import Environment, make_linestring

__all__ = [
    "path_loss_from_ap",
    "rssi_map_for_ap",
    "best_ap_field",
    "fused_ap_field",
    "clip_dbm",
    "linear_normalize",
    "sample_training_data",
]


def path_loss_from_ap(env: Environment, ap_pos: np.ndarray, pos: np.ndarray, n_exp: float = 3.0) -> float:
    """Log-distance path loss plus material attenuation for crossed entities."""
    line = make_linestring(ap_pos, pos)
    dist = float(np.linalg.norm(pos - ap_pos))
    if dist == 0.0:
        return 0.0
    pl = 10.0 * n_exp * np.log10(dist)
    for wall in env.walls:
        if line.crosses(wall["line"]):
            pl += env.material_attenuation.get(wall["material"], 0.0)
    for obj in env.objects:
        if line.crosses(obj["box"]):
            pl += env.material_attenuation.get(obj["material"], 0.0)
    return pl


def rssi_map_for_ap(env: Environment, ap_pos: np.ndarray, pts: np.ndarray, P_tx: float = -30.0, n_exp: float = 3.0) -> np.ndarray:
    rssi = np.empty(len(pts), dtype=np.float64)
    for i, p in enumerate(pts):
        rssi[i] = P_tx - path_loss_from_ap(env, ap_pos, p, n_exp=n_exp)
    return rssi


def best_ap_field(env: Environment, positions: np.ndarray, P_tx: float = -30.0, n_exp: float = 3.0) -> np.ndarray:
    """Max-over-APs RSSI field (dBm)."""
    per_ap = [rssi_map_for_ap(env, ap, positions, P_tx=P_tx, n_exp=n_exp) for ap in env.ap_list]
    return np.vstack(per_ap).max(axis=0)


def fused_ap_field(
    env: Environment,
    positions: np.ndarray,
    P_tx: float = -30.0,
    n_exp: float = 3.0,
    mode: str = "power_sum",
    beta: float = 1.0,
) -> np.ndarray:
    """Fuse per-AP RSSI fields (``power_sum``, ``max``, ``avg_dbm`` or ``lse``)."""
    per_ap = np.vstack([rssi_map_for_ap(env, ap, positions, P_tx=P_tx, n_exp=n_exp) for ap in env.ap_list])
    if mode == "power_sum":
        return 10.0 * np.log10(np.sum(10.0 ** (per_ap / 10.0), axis=0))
    if mode == "max":
        return per_ap.max(axis=0)
    if mode == "avg_dbm":
        return per_ap.mean(axis=0)
    if mode == "lse":
        return 10.0 * beta * np.log10(np.sum(10.0 ** (per_ap / (10.0 * beta)), axis=0))
    raise ValueError(f"Unknown fusion mode: {mode}")


def clip_dbm(arr: np.ndarray, lo: float = -50.0, hi: float = 0.0) -> np.ndarray:
    """Min-max rescale an RSSI field into ``[lo, hi]`` for GP targets / plotting."""
    arr = np.asarray(arr, dtype=float)
    vmin, vmax = arr.min(), arr.max()
    if vmax == vmin:
        return np.full_like(arr, lo, dtype=float)
    return (arr - vmin) / (vmax - vmin) * (hi - lo) + lo


def linear_normalize(
    arr: np.ndarray,
    src_lo: float = -100.0,
    src_hi: float = -30.0,
    dst_lo: float = 0.0,
    dst_hi: float = -50.0,
    do_clip: bool = True,
) -> np.ndarray:
    """Linearly map ``[src_lo, src_hi] -> [dst_lo, dst_hi]``."""
    if do_clip:
        arr = np.clip(arr, min(src_lo, src_hi), max(src_lo, src_hi))
    denom = src_hi - src_lo
    if denom == 0:
        return np.full_like(arr, (dst_lo + dst_hi) / 2.0)
    t = (arr - src_lo) / denom
    return dst_lo + t * (dst_hi - dst_lo)


def sample_training_data(
    positions: np.ndarray,
    y_true: np.ndarray,
    num_samples: int = 100,
    noise_std: float = 10.0,
    seed: Optional[int] = None,
):
    """Draw ``num_samples`` noisy measurements from a ground-truth field."""
    if seed is None:
        idx = np.random.choice(len(positions), size=num_samples, replace=False)
        noise = np.random.normal(0.0, noise_std, size=num_samples)
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(positions), size=num_samples, replace=False)
        noise = rng.normal(0.0, noise_std, size=num_samples)
    return positions[idx], y_true[idx] + noise, idx
