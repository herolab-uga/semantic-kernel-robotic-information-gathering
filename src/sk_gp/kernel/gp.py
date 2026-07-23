"""GP regression on the Semantic Kernel (Eq. 8-10).

Standard zero-mean GP regression with the non-stationary semantic kernel
``K'``.  Given training locations ``X`` and observations ``y``, the posterior at a
test location ``x*`` is

    mu[x*]        = k*^T K_y^{-1} y                                   (Eq. 8)
    sigma2[x*]    = k** - k*^T K_y^{-1} k*                            (Eq. 9)
    sigma2_y[x*]  = sigma2[x*] + sigma2_n(x*)                         (Eq. 10)

with ``K_y = K'(X, X)``, ``k* = K'(X, x*)`` and ``k** = K(x*, x*)``.  Inference is
delegated to scikit-learn's :class:`GaussianProcessRegressor` with
``optimizer=None`` (semantic hyperparameters are supplied by the Scene-LLM, not
fit to the data).
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from .._release import require_scene_llm_parameters
from .hyperfield import SemanticHyperField
from .semantic_kernel import SemanticGibbsRBF, SemanticKernel

__all__ = [
    "fit_predict_stationary",
    "fit_predict_semantic",
    "fit_predict_semantic_kernel",
]


def fit_predict_stationary(
    X_train: np.ndarray,
    y_train: np.ndarray,
    positions: np.ndarray,
    lengthscale: float = 1.0,
    alpha: float = 1e-3,
    normalize_y: bool = True,
) -> Tuple[np.ndarray, np.ndarray, GaussianProcessRegressor]:
    """Stationary RBF-GP baseline."""
    kernel = C(1.0, constant_value_bounds="fixed") * RBF(length_scale=lengthscale, length_scale_bounds="fixed")
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=normalize_y, optimizer=None)
    gp.fit(X_train, y_train)
    y_pred, sigma = gp.predict(positions, return_std=True)
    return y_pred, sigma, gp


def fit_predict_semantic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    positions: np.ndarray,
    semantic_l_fn,
    variance: float = 1.0,
    jitter: float = 3e-5,
    alpha: float = 1e-2,
    normalize_y: bool = True,
) -> Tuple[np.ndarray, np.ndarray, GaussianProcessRegressor]:
    """Lengthscale-only semantic GP (``SemanticGibbsRBF``)."""
    kernel = SemanticGibbsRBF(lengthscale_fn=semantic_l_fn, variance=variance, jitter=jitter)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=normalize_y, optimizer=None)
    gp.fit(X_train, y_train)
    y_pred, sigma = gp.predict(positions, return_std=True)
    return y_pred, sigma, gp


def fit_predict_semantic_kernel(
    X_train: np.ndarray,
    y_train: np.ndarray,
    positions: np.ndarray,
    hyperfield: SemanticHyperField,
    gate_entities: List[dict],
    gate_distance_scale: float,
    variance: float = 1.0,
    jitter: float = 3e-5,
    alpha: float = 0.0,
    normalize_y: bool = True,
    use_semantic_gate: bool = True,
    use_geometry_gate: bool = True,
    noise_field=None,
    scene_llm_source=None,
    allow_untuned: bool = False,
) -> Tuple[np.ndarray, np.ndarray, GaussianProcessRegressor]:
    """Fit and predict with the full Semantic Kernel ``K'`` (Eq. 7-10).

    Requires a Scene-LLM parameter source (``scene_llm_source=...`` — a few-shot or
    fine-tuned config, or an adapter path). Without one the call raises
    ``SceneLLMParametersRequired`` unless ``allow_untuned=True`` is set. See
    :mod:`sk_gp._release`.

    The GP is refit with an increasing diagonal load if Cholesky fails, which can
    happen for aggressive (near-degenerate) LLM lengthscale proposals.
    """
    require_scene_llm_parameters(scene_llm_source, allow_untuned=allow_untuned)
    kernel = SemanticKernel(
        hyperfield=hyperfield,
        gate_entities=gate_entities,
        gate_distance_scale=gate_distance_scale,
        variance=variance,
        jitter=jitter,
        use_semantic_gate=use_semantic_gate,
        use_geometry_gate=use_geometry_gate,
        noise_field=noise_field,
    )
    hyperfield.fit_centers(X_train, mode=hyperfield.center_mode)

    alpha_try = float(max(alpha, 0.0))
    gp: GaussianProcessRegressor | None = None
    last_error: np.linalg.LinAlgError | None = None
    for _ in range(6):
        gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha_try, normalize_y=normalize_y, optimizer=None)
        try:
            gp.fit(X_train, y_train)
            break
        except np.linalg.LinAlgError as exc:
            last_error = exc
            alpha_floor = max(float(jitter), 1e-8)
            alpha_try = max(alpha_floor, alpha_try * 10.0 if alpha_try > 0.0 else alpha_floor)
    else:
        assert last_error is not None
        raise last_error

    assert gp is not None
    y_pred, sigma = gp.predict(positions, return_std=True)
    return y_pred, sigma, gp
