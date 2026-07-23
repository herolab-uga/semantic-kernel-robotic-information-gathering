"""The Semantic Kernel ``K'`` (Eq. 7).

    K'(x, x') = G(x, x') * K(x, x'; l(.), sigma2(.)) + sigma2_n(x) * delta(x, x')

where the base kernel ``K`` is the non-stationary Gibbs generalization of the RBF
kernel with a spatially varying lengthscale ``l(.)`` and signal variance
``sigma2(.)``:

    K(x, x') = sigma2 * ( 2 l(x) l(x') / (l(x)^2 + l(x')^2) )^{d/2}
                       * exp( -||x - x'||^2 / (l(x)^2 + l(x')^2) )

``G`` is the semantic gate (Eq. 3-4) combining soft co-membership
``z(x)^T z(x')`` with geometry-aware, signed-distance attenuation.  A spatially
varying noise field ``sigma2_n(x)`` (optional) is added on the diagonal.

Three kernels are exported, in increasing expressiveness:

* :class:`SemanticGibbsRBF`    -- lengthscale field only.
* :class:`SemanticGatedGibbs`  -- lengthscale + semantic co-membership gate.
* :class:`SemanticKernel`      -- full model: lengthscale + membership +
  geometry-aware gate + optional signal/noise variance fields (paper's ``K'``).

All three subclass :class:`sklearn.gaussian_process.kernels.Kernel` and are used
with ``optimizer=None`` (hyperparameters come from the Scene-LLM, not from
marginal-likelihood optimization on noisy observations).
"""

from __future__ import annotations

import math
from typing import Any, List, Optional

import numpy as np
from sklearn.gaussian_process.kernels import Kernel

from .gate import signed_distance_to_box, signed_distance_to_wall
from .hyperfield import SemanticHyperField

__all__ = ["SemanticGibbsRBF", "SemanticGatedGibbs", "SemanticKernel"]

_EPS = 1e-12


def _sq_dists(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X_norm = np.sum(X * X, axis=1)[:, None]
    Y_norm = np.sum(Y * Y, axis=1)[None, :]
    return np.maximum(X_norm + Y_norm - 2.0 * (X @ Y.T), 0.0)


def _gibbs_base(lX: np.ndarray, lY: np.ndarray, D2: np.ndarray, variance: float, dim: float) -> np.ndarray:
    """Non-stationary Gibbs/RBF base kernel with spatially varying lengthscales."""
    l2sum = np.maximum(lX * lX + lY * lY, _EPS)
    base_ratio = np.maximum((2.0 * lX * lY) / l2sum, _EPS)
    pref = np.power(base_ratio, 0.5 * dim)
    return (variance * pref) * np.exp(-D2 / l2sum)


def _stabilize_train_gram(K: np.ndarray, jitter: float) -> np.ndarray:
    """Symmetrize and adaptively load the diagonal until Cholesky succeeds."""
    K = 0.5 * (K + K.T)
    idx = np.diag_indices_from(K)
    K[idx] += jitter
    try:
        np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        shift = jitter
        for _ in range(6):
            try:
                np.linalg.cholesky(K + np.eye(K.shape[0]) * shift)
                K[idx] += shift
                break
            except np.linalg.LinAlgError:
                shift *= 10.0
    return K


class SemanticGibbsRBF(Kernel):
    """Non-stationary Gibbs RBF using only the lengthscale field ``l(x)``."""

    def __init__(self, lengthscale_fn, variance: float = 1.0, jitter: float = 3e-5):
        self.lengthscale_fn = lengthscale_fn
        self.variance = float(variance)
        self.jitter = float(jitter)

    def __repr__(self):
        return f"SemanticGibbsRBF(variance={self.variance}, jitter={self.jitter})"

    def get_params(self, deep=True):
        return {"lengthscale_fn": self.lengthscale_fn, "variance": self.variance, "jitter": self.jitter}

    def set_params(self, **params):
        for key in ("lengthscale_fn", "variance", "jitter"):
            if key in params:
                setattr(self, key, params[key] if key == "lengthscale_fn" else float(params[key]))
        return self

    def __call__(self, X, Y=None, eval_gradient=False):
        if eval_gradient:
            raise ValueError("Gradient not implemented; use optimizer=None in GPR.")
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Y = X if Y is None else np.atleast_2d(np.asarray(Y, dtype=np.float64))
        lX = np.asarray(self.lengthscale_fn(X), dtype=np.float64).reshape(-1, 1)
        lY = np.asarray(self.lengthscale_fn(Y), dtype=np.float64).reshape(1, -1)
        K = _gibbs_base(lX, lY, _sq_dists(X, Y), self.variance, float(X.shape[1]))
        if (Y is X) or (X.shape == Y.shape and np.allclose(X, Y)):
            K = _stabilize_train_gram(K, self.jitter)
        return K

    def diag(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return np.full(X.shape[0], self.variance, dtype=np.float64)

    def is_stationary(self):
        return False


class SemanticGatedGibbs(Kernel):
    """Gibbs RBF gated by soft semantic co-membership ``z(x)^T z(x')`` (Eq. 4)."""

    def __init__(self, hyperfield: SemanticHyperField, variance: float = 1.0, jitter: float = 3e-5):
        self.hyperfield = hyperfield
        self.variance = float(variance)
        self.jitter = float(jitter)

    def __repr__(self):
        return f"SemanticGatedGibbs(variance={self.variance}, jitter={self.jitter}, K={self.hyperfield.K})"

    def get_params(self, deep=True):
        return {"hyperfield": self.hyperfield, "variance": self.variance, "jitter": self.jitter}

    def set_params(self, **params):
        for key in ("hyperfield", "variance", "jitter"):
            if key in params:
                setattr(self, key, params[key] if key == "hyperfield" else float(params[key]))
        return self

    def __call__(self, X, Y=None, eval_gradient=False):
        if eval_gradient:
            raise ValueError("Gradient not implemented; use optimizer=None in GPR.")
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Y = X if Y is None else np.atleast_2d(np.asarray(Y, dtype=np.float64))
        lX, zX = self.hyperfield(X)
        lY, zY = self.hyperfield(Y)
        lX = lX.reshape(-1, 1)
        lY = lY.reshape(1, -1)
        gate = zX @ zY.T
        K = gate * _gibbs_base(lX, lY, _sq_dists(X, Y), self.variance, float(X.shape[1]))
        if (Y is X) or (X.shape == Y.shape and np.allclose(X, Y)):
            K = _stabilize_train_gram(K, self.jitter)
        return K

    def diag(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return np.full(X.shape[0], self.variance, dtype=np.float64)

    def is_stationary(self):
        return False


class SemanticKernel(Kernel):
    """Full Semantic Kernel ``K'`` (Eq. 7).

    Combines a non-stationary Gibbs base kernel with:

    * the semantic co-membership gate ``z(x)^T z(x')`` (Eq. 4, toggled by
      ``use_semantic_gate``),
    * the geometry-aware, signed-distance attenuation ``A(x, x')`` (Eq. 3, toggled
      by ``use_geometry_gate``),
    * an optional spatially varying noise field ``sigma2_n(x)`` added on the
      diagonal (``noise_field``); when ``None`` the noise is handled by the GP's
      ``alpha`` / an additive ``WhiteKernel``.
    """

    def __init__(
        self,
        hyperfield: SemanticHyperField,
        gate_entities: List[dict],
        gate_distance_scale: float,
        variance: float = 1.0,
        jitter: float = 3e-5,
        use_semantic_gate: bool = True,
        use_geometry_gate: bool = True,
        noise_field=None,
    ):
        self.hyperfield = hyperfield
        self.gate_entities = gate_entities
        self.gate_distance_scale = float(max(gate_distance_scale, 1e-6))
        self.variance = float(variance)
        self.jitter = float(jitter)
        self.use_semantic_gate = bool(use_semantic_gate)
        self.use_geometry_gate = bool(use_geometry_gate)
        self.noise_field = noise_field
        self._gate_blend_width = 1.75

    def __repr__(self):
        return (
            "SemanticKernel("
            f"variance={self.variance}, jitter={self.jitter}, "
            f"entities={len(self.gate_entities)}, "
            f"semantic_gate={self.use_semantic_gate}, geometry_gate={self.use_geometry_gate})"
        )

    def get_params(self, deep=True):
        return {
            "hyperfield": self.hyperfield,
            "gate_entities": self.gate_entities,
            "gate_distance_scale": self.gate_distance_scale,
            "variance": self.variance,
            "jitter": self.jitter,
            "use_semantic_gate": self.use_semantic_gate,
            "use_geometry_gate": self.use_geometry_gate,
            "noise_field": self.noise_field,
        }

    def set_params(self, **params):
        for key, value in params.items():
            if key == "gate_distance_scale":
                self.gate_distance_scale = float(max(value, 1e-6))
            elif key in ("variance", "jitter"):
                setattr(self, key, float(value))
            elif key in ("use_semantic_gate", "use_geometry_gate"):
                setattr(self, key, bool(value))
            elif key in ("hyperfield", "gate_entities", "noise_field"):
                setattr(self, key, value)
        return self

    # -- gate building blocks (Eq. 3) -------------------------------------
    @staticmethod
    def _sigmoid(u: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(u, -60.0, 60.0)))

    @staticmethod
    def _smootherstep01(u: np.ndarray) -> np.ndarray:
        u = np.clip(u, 0.0, 1.0)
        return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)

    def _signed_distance(self, points: np.ndarray, entity: dict) -> np.ndarray:
        if entity["kind"] == "wall":
            return signed_distance_to_wall(points, entity["p0"], entity["p1"])
        return signed_distance_to_box(points, entity["bounds"])

    def _effective_gamma(self, entity: dict) -> float:
        return math.sqrt(max(float(entity["gamma"]), _EPS))

    def _soft_side_probability(self, points: np.ndarray, entity: dict) -> np.ndarray:
        signed = self._signed_distance(points, entity) / self.gate_distance_scale
        gamma = self._effective_gamma(entity)
        return self._sigmoid((gamma * signed) / self._gate_blend_width)

    def _entity_gate_profile(self, points: np.ndarray, entity: dict) -> np.ndarray:
        signed = self._signed_distance(points, entity)
        gamma = self._effective_gamma(entity)
        distance_norm = (gamma * np.abs(signed)) / (self.gate_distance_scale * self._gate_blend_width)
        return 1.0 - self._smootherstep01(distance_norm)

    @staticmethod
    def _semantic_similarity(zX: np.ndarray, zY: np.ndarray) -> np.ndarray:
        zX = np.asarray(zX, dtype=np.float64)
        zY = np.asarray(zY, dtype=np.float64)
        zX_norm = zX / np.maximum(np.linalg.norm(zX, axis=1, keepdims=True), _EPS)
        zY_norm = zY / np.maximum(np.linalg.norm(zY, axis=1, keepdims=True), _EPS)
        return np.clip(zX_norm @ zY_norm.T, 0.0, 1.0)

    def _geometry_attenuation(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Geometry-aware attenuation ``A(x, x')`` of Eq. (3)."""
        if (not self.use_geometry_gate) or (not self.gate_entities):
            return np.ones((X.shape[0], Y.shape[0]), dtype=np.float64)

        attenuation = np.ones((X.shape[0], Y.shape[0]), dtype=np.float64)
        for entity in self.gate_entities:
            lam = float(entity["lambda"])
            if lam <= 0.0:
                continue
            pX = self._soft_side_probability(X, entity)
            pY = self._soft_side_probability(Y, entity)
            min_cross_corr = float(np.clip(math.exp(-lam), _EPS, 1.0))
            delta = math.acos(min_cross_corr)
            thetaX = delta * (1.0 - pX)
            thetaY = delta * (1.0 - pY)
            local_attn = np.cos(thetaX[:, None] - thetaY[None, :])
            attenuation *= np.clip(local_attn, min_cross_corr, 1.0)
        return attenuation

    def local_gate_factor(self, X: np.ndarray) -> np.ndarray:
        """Per-location gate strength in ``[0, 1]`` (for lengthscale visualization)."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        factor = np.ones(X.shape[0], dtype=np.float64)
        if self.use_geometry_gate:
            for entity in self.gate_entities:
                lam = float(entity["lambda"])
                if lam <= 0.0:
                    continue
                influence = self._entity_gate_profile(X, entity)
                factor = np.minimum(factor, 1.0 - (1.0 - math.exp(-lam)) * influence)
        return np.clip(factor, 0.0, 1.0)

    # -- kernel evaluation (Eq. 7) ----------------------------------------
    def __call__(self, X, Y=None, eval_gradient=False):
        if eval_gradient:
            raise ValueError("Gradient not implemented; use optimizer=None in GPR.")
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Y = X if Y is None else np.atleast_2d(np.asarray(Y, dtype=np.float64))

        lX, zX = self.hyperfield(X)
        lY, zY = self.hyperfield(Y)
        lX = np.asarray(lX, dtype=np.float64).reshape(-1, 1)
        lY = np.asarray(lY, dtype=np.float64).reshape(1, -1)

        gate = np.ones((X.shape[0], Y.shape[0]), dtype=np.float64)
        if self.use_semantic_gate:
            gate *= self._semantic_similarity(zX, zY)
        gate *= self._geometry_attenuation(X, Y)

        K = gate * _gibbs_base(lX, lY, _sq_dists(X, Y), self.variance, float(X.shape[1]))

        is_train_gram = (Y is X) or (X.shape == Y.shape and np.allclose(X, Y))
        if is_train_gram:
            K = _stabilize_train_gram(K, self.jitter)
            if self.noise_field is not None:
                # Add the spatially varying noise field sigma2_n(x) on the diagonal.
                K[np.diag_indices_from(K)] += np.asarray(self.noise_field(X), dtype=np.float64).reshape(-1)
        return K

    def diag(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        d = np.full(X.shape[0], self.variance, dtype=np.float64)
        if self.noise_field is not None:
            d = d + np.asarray(self.noise_field(X), dtype=np.float64).reshape(-1)
        return d

    def is_stationary(self):
        return False
