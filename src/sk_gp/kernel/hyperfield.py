"""Semantic hyper-field: ``H(x) = {l(x), z(x)}``.

The hyper-field couples a non-stationary lengthscale ``l(x)`` with a soft
membership vector ``z(x) in [0, 1]^{M_i}`` (``sum_m z_m(x) = 1``).  The membership
is the semantic co-grouping term used inside the semantic gate ``G`` (Eq. 4):
locations with similar memberships remain correlated, while locations that fall
in different semantic groups are attenuated.

Memberships are obtained by softly binning ``l(x)`` into ``K`` groups with
Gaussian bumps in lengthscale space, optionally biased by LLM priors ``pi`` (added
as ``log pi`` to the logits).  Group centers can be placed linearly over
``[l_min, l_max]`` or fit to the quantiles of ``l(x)`` over a reference set.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .fields import SemanticLengthscale

__all__ = ["SemanticHyperField"]


class SemanticHyperField:
    def __init__(
        self,
        sem_l_fn,
        K: int = 3,
        temperature: float = 1.0,
        center_mode: str = "linear",
        center_spread: Optional[float] = None,
        priors: Optional[Sequence[float]] = None,
        cache_round: int = 3,
    ):
        self.sem_l_fn = sem_l_fn
        self.K = int(K)
        if self.K < 2:
            raise ValueError("K (number of semantic groups) must be >= 2.")
        self.temperature = float(max(1e-6, temperature))
        self.center_mode = str(center_mode)
        self.cache_round = int(cache_round)

        l_min = getattr(sem_l_fn, "l_min", 0.35)
        l_max = getattr(sem_l_fn, "l_max", 1.40)
        self.centers = np.linspace(l_min, l_max, self.K, dtype=np.float64)

        if center_spread is None:
            self.sigma_l = float((l_max - l_min) / (2.0 * self.K + 1e-9))
        else:
            self.sigma_l = float(max(1e-9, center_spread))

        if priors is None:
            self.priors = np.ones(self.K, dtype=np.float64)
        else:
            self.set_priors(priors)

    def set_priors(self, priors: Sequence[float]) -> None:
        priors = np.asarray(priors, dtype=np.float64).reshape(-1)
        if priors.shape[0] != self.K:
            raise ValueError(f"priors must have length K={self.K}")
        self.priors = np.maximum(priors, 1e-12)

    def fit_centers(self, X_ref: np.ndarray, mode: str = "quantile") -> None:
        """Adapt group centers to a reference set (typically the training inputs)."""
        X_ref = np.atleast_2d(np.asarray(X_ref, dtype=np.float64))
        if X_ref.size == 0:
            return
        lvals = np.asarray(self.sem_l_fn(X_ref), dtype=np.float64).reshape(-1)
        if mode == "quantile":
            qs = np.linspace(0.0, 1.0, self.K) * 0.9 + 0.05
            self.centers = np.quantile(lvals, qs)
        elif mode == "linear":
            self.centers = np.linspace(np.min(lvals), np.max(lvals), self.K, dtype=np.float64)
        else:
            raise ValueError("mode must be 'quantile' or 'linear'")

    def lengthscale(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.sem_l_fn(X), dtype=np.float64).reshape(-1)

    def membership(self, X: np.ndarray) -> np.ndarray:
        """Soft membership ``z(x)`` via a temperature-scaled softmax over group bumps."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        lvals = self.lengthscale(X)
        diffs = (lvals[:, None] - self.centers[None, :]) / (self.sigma_l + 1e-12)
        logits = -0.5 * (diffs ** 2) / self.temperature
        logits += np.log(self.priors[None, :])
        logits -= logits.max(axis=1, keepdims=True)
        z = np.exp(logits)
        z /= np.sum(z, axis=1, keepdims=True) + 1e-12
        return z

    def __call__(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.lengthscale(X), self.membership(X)
