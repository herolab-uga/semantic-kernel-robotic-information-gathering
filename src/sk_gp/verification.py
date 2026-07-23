"""Formal verification of Scene-LLM parameters (Theorem 1).

The Scene-LLM emits a discrete parameter set ``Gamma`` that must be checked before
it is used to instantiate the Semantic Kernel.  Theorem 1 accepts a
parameterization ``Gamma_v`` only if it lies in ``Theta_adm cap C``:

* ``Theta_adm`` -- boundedness / admissibility constraints on all four
  non-stationarities (lengthscale bounds and ordering, positive smoothness and
  radius, non-negative variance and noise, non-negative gate attenuation and
  positive sharpness, valid group count and priors).
* ``C`` -- physical feasibility: primitive centers and scene entities lie inside
  the spatial domain ``Omega`` (checked when domain bounds are supplied).

If verification fails, a **structured failure log** is returned so the caller can
either re-invoke the Scene-LLM with corrective feedback (up to a retry limit) or
fall back to a stationary kernel -- guaranteeing that ``K'`` is always instantiated
from verified parameters and therefore yields a finite, positive-semidefinite
covariance matrix and a well-defined GP posterior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["AdmissibleBounds", "VerificationResult", "verify_parameters", "FormalVerifier"]


@dataclass
class AdmissibleBounds:
    """Boundedness constraints defining ``Theta_adm``."""

    l_lower: float = 1e-3
    l_upper: float = 1e3
    tau_lower: float = 1e-3
    radius_lower: float = 1e-3
    beta_lower: float = 0.0
    variance_lower: float = 0.0
    noise_lower: float = 0.0
    jitter_lower: float = 1e-9
    jitter_upper: float = 1e-1
    gate_lambda_lower: float = 0.0
    gate_gamma_lower: float = 1e-3
    k_min: int = 2
    k_max: int = 8


@dataclass
class VerificationResult:
    """Outcome of :func:`verify_parameters`."""

    valid: bool
    params: Dict[str, Any]
    failures: List[str] = field(default_factory=list)

    def failure_log(self) -> str:
        return "; ".join(self.failures)


def _num(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def verify_parameters(
    params: Dict[str, Any],
    *,
    bounds: Optional[AdmissibleBounds] = None,
    domain: Optional[Tuple[float, float, float, float]] = None,
    entities: Optional[List[Dict[str, Any]]] = None,
) -> VerificationResult:
    """Check ``params`` (``Gamma``) against ``Theta_adm cap C``.

    Parameters
    ----------
    params : dict
        Parameter set.  Recognized keys include ``l_min``/``l_max`` (or
        ``l_min_ratio``/``l_max_ratio``), ``tau``, ``radius``, ``beta``, ``K``,
        ``priors``, ``sem_variance``/``variance``, ``sem_jitter``/``jitter``,
        ``noise``, ``centers`` and ``gate_material_params``.  Unknown keys are
        ignored, so both the raw LLM config and the ratio config can be verified.
    domain : (xmin, xmax, ymin, ymax), optional
        Spatial domain ``Omega`` for the physical-feasibility check ``C``.
    entities : list of gate entities, optional
        Checked to lie within ``Omega`` when ``domain`` is given.
    """
    bounds = bounds or AdmissibleBounds()
    failures: List[str] = []

    # --- Theta_adm: lengthscale field ------------------------------------
    l_min = _num(params.get("l_min", params.get("l_min_ratio")))
    l_max = _num(params.get("l_max", params.get("l_max_ratio")))
    if l_min is not None and l_max is not None:
        if not (l_min > 0.0):
            failures.append(f"l_min={l_min} must be > 0")
        if not (l_max > l_min):
            failures.append(f"l_max={l_max} must exceed l_min={l_min}")
        if l_min is not None and (l_min < bounds.l_lower or l_min > bounds.l_upper):
            failures.append(f"l_min={l_min} outside [{bounds.l_lower}, {bounds.l_upper}]")

    tau = _num(params.get("tau", params.get("tau_ratio")))
    if tau is not None and tau <= bounds.tau_lower:
        failures.append(f"tau={tau} must be > {bounds.tau_lower}")

    radius = _num(params.get("radius", params.get("radius_ratio")))
    if radius is not None and radius <= bounds.radius_lower:
        failures.append(f"radius={radius} must be > {bounds.radius_lower}")

    beta = _num(params.get("beta"))
    if beta is not None and beta < bounds.beta_lower:
        failures.append(f"beta={beta} must be >= {bounds.beta_lower}")

    # --- Theta_adm: variance / noise fields ------------------------------
    variance = _num(params.get("sem_variance", params.get("variance")))
    if variance is not None and variance <= bounds.variance_lower:
        failures.append(f"variance={variance} must be > {bounds.variance_lower}")

    noise = _num(params.get("noise", params.get("sem_alpha")))
    if noise is not None and noise < bounds.noise_lower:
        failures.append(f"noise={noise} must be >= {bounds.noise_lower}")

    jitter = _num(params.get("sem_jitter", params.get("jitter")))
    if jitter is not None and not (bounds.jitter_lower <= jitter <= bounds.jitter_upper):
        failures.append(f"jitter={jitter} outside [{bounds.jitter_lower}, {bounds.jitter_upper}]")

    # --- Theta_adm: semantic groups / priors -----------------------------
    if "K" in params:
        k = params.get("K")
        if not isinstance(k, int) or not (bounds.k_min <= k <= bounds.k_max):
            failures.append(f"K={k} must be an integer in [{bounds.k_min}, {bounds.k_max}]")
        priors = params.get("priors")
        if priors is not None:
            if len(priors) != (k if isinstance(k, int) else -1):
                failures.append("priors length must equal K")
            elif any(_num(p, -1) < 0 for p in priors):
                failures.append("priors must be non-negative")

    # --- Theta_adm: gate parameters --------------------------------------
    for material, gp in (params.get("gate_material_params") or {}).items():
        lam = _num(gp.get("lambda"))
        gamma = _num(gp.get("gamma"))
        if lam is not None and lam < bounds.gate_lambda_lower:
            failures.append(f"gate lambda[{material}]={lam} must be >= {bounds.gate_lambda_lower}")
        if gamma is not None and gamma <= bounds.gate_gamma_lower:
            failures.append(f"gate gamma[{material}]={gamma} must be > {bounds.gate_gamma_lower}")

    # --- C: physical feasibility -----------------------------------------
    if domain is not None:
        xmin, xmax, ymin, ymax = domain

        def _inside(px, py):
            return (xmin <= px <= xmax) and (ymin <= py <= ymax)

        for center in params.get("centers", []) or []:
            if len(center) >= 2 and not _inside(float(center[0]), float(center[1])):
                failures.append(f"primitive center {tuple(center[:2])} outside domain")
        for ent in entities or []:
            if ent.get("kind") == "wall":
                for pt in (ent.get("p0"), ent.get("p1")):
                    if pt is not None and not _inside(float(pt[0]), float(pt[1])):
                        failures.append("gate entity (wall) endpoint outside domain")
            elif ent.get("kind") == "object" and ent.get("bounds") is not None:
                bx0, by0, bx1, by1 = ent["bounds"]
                cx, cy = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)
                if not _inside(cx, cy):
                    failures.append("gate entity (object) center outside domain")

    return VerificationResult(valid=len(failures) == 0, params=params, failures=failures)


class FormalVerifier:
    """Stateful verifier with retry / fallback bookkeeping (Theorem 1).

    Wraps :func:`verify_parameters` with a retry counter.  ``review`` returns
    ``(verified_params_or_None, result)``; the caller re-invokes the Scene-LLM with
    ``result.failure_log()`` while ``should_retry`` is true, and otherwise applies a
    stationary fallback so a verified kernel is always produced.
    """

    def __init__(
        self,
        bounds: Optional[AdmissibleBounds] = None,
        domain: Optional[Tuple[float, float, float, float]] = None,
        retry_limit: int = 3,
    ):
        self.bounds = bounds or AdmissibleBounds()
        self.domain = domain
        self.retry_limit = int(retry_limit)
        self.attempts = 0
        self.failure_history: List[List[str]] = []

    @property
    def should_retry(self) -> bool:
        return self.attempts < self.retry_limit

    def review(self, params: Dict[str, Any], entities=None) -> Tuple[Optional[Dict[str, Any]], VerificationResult]:
        self.attempts += 1
        result = verify_parameters(params, bounds=self.bounds, domain=self.domain, entities=entities)
        if not result.valid:
            self.failure_history.append(result.failures)
        return (result.params if result.valid else None), result
