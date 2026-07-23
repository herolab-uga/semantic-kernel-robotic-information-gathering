"""Spatially varying non-stationarity fields for the Semantic Kernel.

This module implements the four continuous non-stationarity fields that the
Scene-LLM parameterizes (Section III of the paper):

* Lengthscale field ``l(x)``            -- Eq. (1)-(2)
* Signal variance field ``sigma2(x)``   -- Eq. (5)
* Noise variance field ``sigma2_n(x)``  -- Eq. (5)

All three share the same *convex-combination-of-primitives* construction: each
field is a normalized blend of ``M`` primitive values, where the (unnormalized)
weight of primitive ``m`` at location ``x`` decays with distance from its center
of influence ``c_m`` (Eq. 2):

    w~_m(x) = theta_m * exp(-||x - c_m||^2 / (2 tau_m^2)) * 1[||x - c_m|| <= r_m]
    w_m(x)  = w~_m(x) / sum_j w~_j(x)                                       (Eq. 2)
    f(x)    = sum_m w_m(x) * value_m           (clipped to [v_min, v_max])  (Eq. 1)

A second, material-driven parameterization of the lengthscale
(:class:`SemanticLengthscale`) is also provided.  Instead of explicit primitive
centers, it accumulates attenuation contributions from nearby scene entities
(walls, objects) and maps them to a shorter lengthscale near dense/attenuating
geometry.  This is the form consumed by the Scene-LLM few-shot / QLoRA pipeline
(the LLM emits ``tau, beta, l_min, l_max, radius`` ratios), and it plugs into the
same :class:`~sk_gp.kernel.hyperfield.SemanticHyperField` interface.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional, Sequence

import numpy as np
from shapely.geometry import Point

from ..envs.environment import Environment

__all__ = [
    "primitive_weights",
    "PrimitiveField",
    "LengthscaleField",
    "VarianceField",
    "NoiseField",
    "SemanticLengthscale",
]


def primitive_weights(
    x: np.ndarray,
    centers: np.ndarray,
    tau: np.ndarray,
    radius: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """Normalized convex weights ``w_m(x)`` of Eq. (2).

    Parameters
    ----------
    x : (N, 2) array of query locations.
    centers : (M, 2) array of primitive centers of influence ``c_m``.
    tau : (M,) smoothness scales ``tau_m > 0``.
    radius : (M,) hard cut-off influence radii ``r_m > 0``.
    theta : (M,) contrast scales ``theta_m > 0``.

    Returns
    -------
    (N, M) array of weights, each row summing to 1.  If every primitive is
    outside its influence radius at a location, weights fall back to a uniform
    distribution so the blended field remains well defined.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    centers = np.atleast_2d(np.asarray(centers, dtype=np.float64))
    tau = np.asarray(tau, dtype=np.float64).reshape(-1)
    radius = np.asarray(radius, dtype=np.float64).reshape(-1)
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)

    # Pairwise distances ||x - c_m||, shape (N, M).
    diff = x[:, None, :] - centers[None, :, :]
    dist = np.sqrt(np.maximum(np.sum(diff * diff, axis=2), 0.0))

    inside = dist <= radius[None, :]
    unnorm = theta[None, :] * np.exp(-(dist ** 2) / (2.0 * (tau[None, :] ** 2) + 1e-12))
    unnorm = np.where(inside, unnorm, 0.0)

    denom = unnorm.sum(axis=1, keepdims=True)
    # Locations with no active primitive -> uniform fallback.
    empty = denom[:, 0] <= 1e-12
    weights = np.where(denom > 1e-12, unnorm / (denom + 1e-12), 1.0 / centers.shape[0])
    if np.any(empty):
        weights[empty] = 1.0 / centers.shape[0]
    return weights


class PrimitiveField:
    """A continuous field built from ``M`` primitives (Eq. 1 / Eq. 5).

    ``values`` holds the ``M`` primitive scalars that are blended by the
    distance-decayed convex weights of :func:`primitive_weights`.  The output is
    clipped to ``[v_min, v_max]`` so the field stays inside the admissible set
    ``Theta_adm`` used by the formal verifier (Theorem 1).
    """

    def __init__(
        self,
        centers: Sequence[Sequence[float]],
        values: Sequence[float],
        tau: Sequence[float] | float,
        radius: Sequence[float] | float,
        theta: Optional[Sequence[float]] = None,
        v_min: float = -np.inf,
        v_max: float = np.inf,
    ):
        self.centers = np.atleast_2d(np.asarray(centers, dtype=np.float64))
        self.values = np.asarray(values, dtype=np.float64).reshape(-1)
        m = self.centers.shape[0]
        if self.values.shape[0] != m:
            raise ValueError("Number of primitive values must match number of centers.")
        self.tau = self._broadcast(tau, m)
        self.radius = self._broadcast(radius, m)
        self.theta = np.ones(m) if theta is None else self._broadcast(theta, m)
        self.v_min = float(v_min)
        self.v_max = float(v_max)

    @staticmethod
    def _broadcast(value, m: int) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size == 1:
            arr = np.repeat(arr, m)
        if arr.size != m:
            raise ValueError(f"Expected scalar or length-{m} parameter, got {arr.size}.")
        return arr

    def weights(self, x: np.ndarray) -> np.ndarray:
        """Convex weights ``w_m(x)`` (Eq. 2) at ``x``."""
        return primitive_weights(x, self.centers, self.tau, self.radius, self.theta)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        w = self.weights(x)
        field = w @ self.values
        return np.clip(field, self.v_min, self.v_max)


class LengthscaleField(PrimitiveField):
    """Lengthscale field ``l(x)`` (Eq. 1-2), clipped to ``[l_min, l_max]``."""

    def __init__(self, centers, lengthscales, tau, radius, theta=None,
                 l_min: float = 0.35, l_max: float = 1.40):
        super().__init__(centers, lengthscales, tau, radius, theta, v_min=l_min, v_max=l_max)
        self.l_min = float(l_min)
        self.l_max = float(l_max)


class VarianceField(PrimitiveField):
    """Signal variance field ``sigma2(x)`` (Eq. 5), non-negative."""

    def __init__(self, centers, variances, tau, radius, theta=None, v_min: float = 1e-6, v_max: float = np.inf):
        super().__init__(centers, variances, tau, radius, theta, v_min=v_min, v_max=v_max)


class NoiseField(PrimitiveField):
    """Measurement-noise variance field ``sigma2_n(x)`` (Eq. 5), non-negative."""

    def __init__(self, centers, noise_variances, tau, radius, theta=None, v_min: float = 0.0, v_max: float = np.inf):
        super().__init__(centers, noise_variances, tau, radius, theta, v_min=v_min, v_max=v_max)


class SemanticLengthscale:
    """Material-driven lengthscale field ``l(x)`` (Scene-LLM pipeline form).

    Rather than explicit primitive centers, the (unnormalized) semantic score
    ``S(x)`` accumulates material attenuation from every wall/object within
    ``radius`` of ``x``, weighted by a Gaussian in distance::

        S(x)  = sum_{e in walls U objects} exp(-(d_e(x)/tau)^2) * atten(material_e)
        l(x)  = clip( l_max / (1 + beta * S(x)),  l_min,  l_max )

    The lengthscale is therefore short near dense/high-attenuation geometry and
    long in open space -- the inductive bias the paper attributes to scene
    semantics.  The Scene-LLM emits ``(tau, beta, l_min, l_max, radius)`` (as
    fractions of the scene scale) that parameterize this field.
    """

    def __init__(
        self,
        env: Environment,
        tau: float = 0.75,
        beta: float = 0.25,
        l_min: float = 0.35,
        l_max: float = 1.40,
        radius: float = 2.0,
        cache_round: int = 3,
    ):
        self.env = env
        self.tau = float(tau)
        self.beta = float(beta)
        self.l_min = float(l_min)
        self.l_max = float(l_max)
        self.radius = float(radius)
        self.cache_round = int(cache_round)

    @lru_cache(maxsize=200_000)
    def _lengthscale_cached(self, x_round: float, y_round: float) -> float:
        pt = Point(float(x_round), float(y_round))
        score = 0.0
        for wall in self.env.walls:
            d = pt.distance(wall["line"])
            if d <= self.radius:
                score += math.exp(-((d / self.tau) ** 2)) * self.env.material_attenuation.get(wall["material"], 0.0)
        for obj in self.env.objects:
            d = pt.distance(obj["box"])
            if d <= self.radius:
                score += math.exp(-((d / self.tau) ** 2)) * self.env.material_attenuation.get(obj["material"], 0.0)
        length = self.l_max / (1.0 + self.beta * score)
        return float(np.clip(length, self.l_min, self.l_max))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            xr, yr = round(x[0], self.cache_round), round(x[1], self.cache_round)
            return self._lengthscale_cached(xr, yr)
        if x.ndim == 2:
            out = np.empty(x.shape[0], dtype=np.float64)
            for i, p in enumerate(x):
                xr, yr = round(float(p[0]), self.cache_round), round(float(p[1]), self.cache_round)
                out[i] = self._lengthscale_cached(xr, yr)
            return out
        raise ValueError("x must have shape (2,) or (N, 2).")
