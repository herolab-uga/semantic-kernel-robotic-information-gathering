"""Temporal outlier detection over Scene-LLM parameter fields (Eq. 6).

Scene-LLM outputs can occasionally violate temporal consistency (hallucinated
jumps).  To keep GP predictions stable we run a lightweight PCA-based outlier
detector over the recent history of verified parameter fields.

At time ``t`` the verified spatial fields induced by ``Gamma_t^v`` are vectorized
into ``phi_t in R^d``.  A PCA model is fit on the previous ``window`` steps,
yielding a mean ``phi_bar``, principal basis ``U_k`` and eigenvalues ``Lambda_k``.
The temporal deviation is the PCA Mahalanobis distance::

    a_t   = U_k^T (phi_t - phi_bar)
    d2_t  = a_t^T Lambda_k^{-1} a_t

To avoid rejecting legitimate change caused by newly observed structure, the
threshold adapts with the normalized **semantic information gain** (Eq. 6)::

    g_t = (|E_t| - |E_{t-1}|) / (|E_t| - |E_{t-1}| + 1)  in [0, 1]

``Gamma_t^v`` is declared an outlier iff ``d2_t > tau0 * (1 + g_t)``.  Rejected
parameter sets are replaced by the most recent valid configuration.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["OutlierDecision", "TemporalOutlierDetector"]


@dataclass
class OutlierDecision:
    is_outlier: bool
    distance: float
    threshold: float
    info_gain: float
    accepted: np.ndarray  # phi_t if accepted, else the last valid phi


class TemporalOutlierDetector:
    """PCA Mahalanobis outlier detector with semantic-gain-adaptive threshold."""

    def __init__(
        self,
        window: int = 5,
        tau0: float = 9.0,
        n_components: int = 3,
        var_floor: float = 1e-6,
        rel_var_floor: float = 1e-3,
    ):
        self.window = int(window)
        self.tau0 = float(tau0)
        self.n_components = int(n_components)
        self.var_floor = float(var_floor)
        # Relative floor on principal eigenvalues: guards against near-degenerate
        # directions (tiny eigenvalues) blowing up the Mahalanobis distance.
        self.rel_var_floor = float(rel_var_floor)
        self._history: deque[np.ndarray] = deque(maxlen=self.window)
        self._last_valid: Optional[np.ndarray] = None
        self._prev_n_entities: Optional[int] = None

    @staticmethod
    def semantic_info_gain(n_entities_t: int, n_entities_prev: Optional[int]) -> float:
        """Normalized semantic information gain ``g_t`` (Eq. 6)."""
        if n_entities_prev is None:
            return 0.0
        delta = int(n_entities_t) - int(n_entities_prev)
        delta = max(delta, 0)  # only newly discovered entities relax the threshold
        return float(delta / (delta + 1.0))

    def _mahalanobis(self, phi: np.ndarray) -> float:
        history = np.stack(self._history, axis=0)
        mean = history.mean(axis=0)
        centered = history - mean
        # PCA via SVD of the centered history.
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        n_obs = centered.shape[0]
        eigvals = (s ** 2) / max(n_obs - 1, 1)
        k = min(self.n_components, vt.shape[0])
        basis = vt[:k]                       # (k, d)  -> rows are principal axes U_k^T
        floor = max(self.var_floor, self.rel_var_floor * float(eigvals[:k].max()))
        eigvals_k = np.maximum(eigvals[:k], floor)
        a = basis @ (phi - mean)             # projection a_t = U_k^T (phi - phi_bar)
        return float(np.sum((a ** 2) / eigvals_k))

    def update(self, phi_t: np.ndarray, n_entities: int) -> OutlierDecision:
        """Score ``phi_t`` and return the accepted (possibly substituted) vector."""
        phi_t = np.asarray(phi_t, dtype=np.float64).reshape(-1)
        info_gain = self.semantic_info_gain(n_entities, self._prev_n_entities)
        threshold = self.tau0 * (1.0 + info_gain)

        # Not enough history yet -> accept and seed the buffer.
        if len(self._history) < max(2, self.n_components):
            distance = 0.0
            is_outlier = False
        else:
            distance = self._mahalanobis(phi_t)
            is_outlier = distance > threshold

        if is_outlier and self._last_valid is not None:
            accepted = self._last_valid
        else:
            accepted = phi_t
            self._last_valid = phi_t
            self._history.append(phi_t)
            self._prev_n_entities = int(n_entities)

        return OutlierDecision(
            is_outlier=is_outlier,
            distance=distance,
            threshold=threshold,
            info_gain=info_gain,
            accepted=accepted,
        )

    def reset(self) -> None:
        self._history.clear()
        self._last_valid = None
        self._prev_n_entities = None
