"""Acquisition function for adaptive informative sampling.

The planner scores candidate locations by a weighted combination of posterior
uncertainty (exploration), novelty (distance to already-sampled points) and a
small travel penalty toward the last visited location.  This is the fixed
adaptive planner (FAP / standard A-IPP) used as the common sampling backbone so
all kernels are compared under identical planning (Section IV).
"""

from __future__ import annotations

import numpy as np

__all__ = ["normalize01", "acquisition_score"]


def normalize01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    vmin, vmax = np.min(values), np.max(values)
    if vmax - vmin < 1e-12:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def acquisition_score(
    mu: np.ndarray,
    std: np.ndarray,
    cand_xy: np.ndarray,
    sampled_xy: np.ndarray,
    last_xy: np.ndarray,
    *,
    domain_diag: float,
    distance_weight: float = 1.0,
    w_mean: float = 0.0,
    w_std: float = 0.86,
    w_novelty: float = 0.14,
    w_travel: float = 0.03,
) -> np.ndarray:
    """Acquisition value for each candidate (higher = sample next)."""
    mu_n = normalize01(mu)
    std_n = normalize01(std)

    if sampled_xy.shape[0] > 0:
        dx = cand_xy[:, None, 0] - sampled_xy[None, :, 0]
        dy = cand_xy[:, None, 1] - sampled_xy[None, :, 1]
        min_dist = np.sqrt(np.min(dx * dx + dy * dy, axis=1))
        novelty = np.clip(min_dist / max(domain_diag, 1e-12), 0.0, 1.0)
    else:
        novelty = np.ones(cand_xy.shape[0], dtype=np.float64)

    d_last = np.linalg.norm(cand_xy - last_xy[None, :], axis=1)
    d_last_n = np.clip(d_last / max(domain_diag, 1e-12), 0.0, 1.0)

    return w_mean * mu_n + w_std * std_n + w_novelty * novelty - w_travel * float(distance_weight) * d_last_n
