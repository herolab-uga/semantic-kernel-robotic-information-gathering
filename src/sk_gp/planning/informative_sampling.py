"""Adaptive informative sampling loop (FAP / A-IPP).

A single robot with a bounded per-step travel budget iteratively (1) fits a GP on
the measurements collected so far, (2) scores unvisited candidates with the
:func:`~sk_gp.planning.acquisition.acquisition_score`, (3) moves to the best
reachable candidate, and (4) takes a noisy measurement.  RMSE and mean posterior
uncertainty against the ground-truth field are logged at every step, producing
the convergence curves reported in the paper.

The loop is kernel-agnostic: pass a ``fit_gp`` callable ``(X_train, y_train) ->
fitted GP`` so the same planner drives RBF, AK, DKL or the Semantic Kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np

from .acquisition import acquisition_score

__all__ = ["SamplingResult", "run_informative_sampling"]


@dataclass
class SamplingResult:
    path_idx: List[int]
    X_train: np.ndarray
    y_obs: np.ndarray
    step_rmse: List[float] = field(default_factory=list)
    step_avg_std: List[float] = field(default_factory=list)
    final_mean: np.ndarray | None = None
    final_std: np.ndarray | None = None


def run_informative_sampling(
    positions: np.ndarray,
    y_true: np.ndarray,
    grid_shape: tuple[int, int],
    fit_gp: Callable[[np.ndarray, np.ndarray], object],
    *,
    n_samples: int = 40,
    candidate_stride: int = 3,
    noise_std_db: float = 5.0,
    step_frac: float = 0.18,
    distance_weight: float = 1.0,
    seed: int = 0,
    record_curves: bool = True,
) -> SamplingResult:
    """Run the adaptive informative-sampling loop and return the trace.

    Parameters
    ----------
    positions : (N, 2) grid of query locations (row-major over ``grid_shape``).
    y_true : (N,) ground-truth field at ``positions``.
    grid_shape : (ny, nx) shape used to build the candidate sub-grid.
    fit_gp : callable mapping ``(X_train, y_train)`` to a fitted GP exposing
        ``predict(X, return_std=True)``.
    """
    rng = np.random.default_rng(seed)
    ny, nx = grid_shape

    idx_grid = np.arange(nx * ny, dtype=np.int64).reshape(ny, nx)
    cand_idx = np.unique(idx_grid[::candidate_stride, ::candidate_stride].ravel())

    xmin, xmax = float(positions[:, 0].min()), float(positions[:, 0].max())
    ymin, ymax = float(positions[:, 1].min()), float(positions[:, 1].max())
    domain_diag = float(np.hypot(xmax - xmin, ymax - ymin))
    grid_dx = abs((xmax - xmin) / max(nx - 1, 1))
    grid_dy = abs((ymax - ymin) / max(ny - 1, 1))
    min_move = 1.2 * max(grid_dx, grid_dy)
    max_step_dist = max(step_frac * domain_diag, min_move)
    relax_schedule = (1.0, 1.6, 2.4, np.inf)

    center_xy = np.array([0.5 * (xmin + xmax), 0.5 * (ymin + ymax)])
    center_local = int(np.argmin(np.sum((positions[cand_idx] - center_xy[None, :]) ** 2, axis=1)))
    start_idx = int(cand_idx[center_local])

    sampled_mask = np.zeros(positions.shape[0], dtype=bool)
    path_idx: List[int] = [start_idx]
    sampled_mask[start_idx] = True
    y_obs: List[float] = [float(y_true[start_idx] + rng.normal(0.0, noise_std_db))]

    while len(path_idx) < int(n_samples):
        available = cand_idx[~sampled_mask[cand_idx]]
        if available.size == 0:
            break

        X_train = positions[np.array(path_idx, dtype=np.int64)]
        y_train = np.asarray(y_obs, dtype=np.float64)
        gp = fit_gp(X_train, y_train)
        mu_c, std_c = gp.predict(positions[available], return_std=True)
        score = acquisition_score(
            mu_c, std_c, positions[available], X_train, positions[path_idx[-1]],
            domain_diag=domain_diag, distance_weight=distance_weight,
        )

        d_cur = np.linalg.norm(positions[available] - positions[path_idx[-1]][None, :], axis=1)
        next_idx = None
        for mult in relax_schedule:
            local_mask = np.ones_like(d_cur, dtype=bool) if np.isinf(mult) else (d_cur <= max_step_dist * float(mult))
            if np.any(local_mask):
                next_idx = int(available[int(np.argmax(np.where(local_mask, score, -np.inf)))])
                break
        if next_idx is None:
            next_idx = int(available[int(np.argmax(score))])

        sampled_mask[next_idx] = True
        path_idx.append(next_idx)
        y_obs.append(float(y_true[next_idx] + rng.normal(0.0, noise_std_db)))

    result = SamplingResult(
        path_idx=path_idx,
        X_train=positions[np.array(path_idx, dtype=np.int64)],
        y_obs=np.asarray(y_obs, dtype=np.float64),
    )

    if record_curves:
        for step_n in range(1, len(path_idx) + 1):
            step_x = positions[np.array(path_idx[:step_n], dtype=np.int64)]
            step_y = np.asarray(y_obs[:step_n], dtype=np.float64)
            gp = fit_gp(step_x, step_y)
            mean, std = gp.predict(positions, return_std=True)
            result.step_rmse.append(float(np.sqrt(np.mean((mean - y_true) ** 2))))
            result.step_avg_std.append(float(np.mean(std)))
        result.final_mean = mean
        result.final_std = std

    return result
