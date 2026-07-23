"""Semantic-kernel configuration schema and sanitization.

The Scene-LLM produces a JSON configuration ``Gamma`` describing the kernel's
non-stationarities.  ``sanitize_semantic_cfg`` clamps every field into its
admissible range (a fast, deterministic pre-filter that mirrors ``Theta_adm``;
the full geometric verification lives in :mod:`sk_gp.verification`).

Two representations are used:

* **semantic cfg** -- absolute values (``tau``, ``l_min``, ``l_max`` ... in meters
  once world-scaled) consumed directly by the kernel builders.
* **lengthscale ratios** -- scene-scale-relative fractions (``tau_ratio``,
  ``l_min_ratio`` ...) that the fine-tuned Scene-LLM predicts; these transfer
  across environments of different sizes.  ``lengthscale_ratios_to_semantic_cfg``
  and its inverse convert between the two given a ``scene_scale``.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

__all__ = [
    "safe_float",
    "safe_int",
    "safe_bool",
    "default_semantic_cfg",
    "sanitize_semantic_cfg",
    "build_world_scaled_semantic_cfg",
    "default_lengthscale_ratios",
    "sanitize_lengthscale_ratios",
    "lengthscale_ratios_to_semantic_cfg",
    "semantic_cfg_to_lengthscale_ratios",
]


def safe_float(value, default, *, min_val=None, max_val=None) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if not np.isfinite(out):
        out = float(default)
    if min_val is not None:
        out = max(out, float(min_val))
    if max_val is not None:
        out = min(out, float(max_val))
    return float(out)


def safe_int(value, default, *, min_val=None, max_val=None) -> int:
    try:
        out = int(value)
    except Exception:
        out = int(default)
    if min_val is not None:
        out = max(out, int(min_val))
    if max_val is not None:
        out = min(out, int(max_val))
    return int(out)


def safe_bool(value, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"1", "true", "yes", "y", "t"}:
            return True
        if low in {"0", "false", "no", "n", "f"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return bool(default)


def _sanitize_gate_material_params(raw_value: Any) -> Dict[str, Dict[str, float]]:
    if not isinstance(raw_value, dict):
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for key, value in raw_value.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        out[key.strip().lower()] = {
            "lambda": safe_float(value.get("lambda"), 0.0, min_val=0.0, max_val=2.0),
            "gamma": safe_float(value.get("gamma"), 1.0, min_val=0.05, max_val=12.0),
        }
    return out


def default_semantic_cfg() -> Dict[str, Any]:
    return {
        "K": 3,
        "priors": [0.5, 1.0, 0.7],
        "tau": 0.9,
        "beta": 0.18,
        "l_min": 0.20,
        "l_max": 1.00,
        "radius": 4.0,
        "temperature": 0.85,
        "center_mode": "quantile",
        "sem_variance": 20.0,
        "sem_jitter": 1e-5,
        "sem_alpha": 0.0,
        "gate_material_params": {},
        "normalize_y": True,
        "reasoning_summary": "Fallback defaults for indoor multi-room RSSI smoothing.",
    }


def sanitize_semantic_cfg(raw_cfg: Dict[str, Any] | None) -> Dict[str, Any]:
    """Clamp an LLM-emitted config into the admissible set."""
    cfg = dict(raw_cfg or {})
    k = safe_int(cfg.get("K"), 3, min_val=2, max_val=4)

    priors_raw = cfg.get("priors", [1.0] * k)
    if not isinstance(priors_raw, list) or not priors_raw:
        priors_raw = [1.0] * k
    priors = np.asarray([safe_float(v, 1.0, min_val=1e-6) for v in priors_raw], dtype=np.float64)
    if priors.size < k:
        priors = np.pad(priors, (0, k - priors.size), constant_values=1.0)
    elif priors.size > k:
        priors = priors[:k]
    priors = (priors / max(priors.sum(), 1e-12)).tolist()

    l_min = safe_float(cfg.get("l_min"), 0.20, min_val=0.05, max_val=0.8)
    l_max = safe_float(cfg.get("l_max"), 1.00, min_val=l_min + 1e-6, max_val=1.6)
    center_mode = str(cfg.get("center_mode", "quantile")).strip().lower()
    if center_mode not in {"quantile", "linear"}:
        center_mode = "quantile"

    return {
        "K": k,
        "priors": priors,
        "tau": safe_float(cfg.get("tau"), 0.9, min_val=0.15, max_val=3.0),
        "beta": safe_float(cfg.get("beta"), 0.18, min_val=0.0, max_val=2.5),
        "l_min": l_min,
        "l_max": l_max,
        "radius": safe_float(cfg.get("radius"), 2.0, min_val=0.25, max_val=4.5),
        "temperature": safe_float(cfg.get("temperature"), 0.85, min_val=0.2, max_val=2.0),
        "center_mode": center_mode,
        "sem_variance": safe_float(cfg.get("sem_variance"), 20.0, min_val=1e-6, max_val=100.0),
        "sem_jitter": safe_float(cfg.get("sem_jitter"), 1e-5, min_val=1e-8, max_val=1e-3),
        "sem_alpha": 0.0,
        "gate_material_params": _sanitize_gate_material_params(cfg.get("gate_material_params")),
        "normalize_y": safe_bool(cfg.get("normalize_y"), True),
        "reasoning_summary": str(cfg.get("reasoning_summary", "Semantic priors chosen from the scene geometry."))[:200],
    }


def build_world_scaled_semantic_cfg(sem_cfg: Dict[str, Any], scene_scale: float) -> Dict[str, Any]:
    """Scale the fractional lengthscale bounds by the scene scale (meters)."""
    out = dict(sem_cfg)
    out["l_min"] = safe_float(sem_cfg.get("l_min"), 0.20, min_val=0.05, max_val=1.6) * scene_scale
    out["l_max"] = (
        safe_float(sem_cfg.get("l_max"), 1.00, min_val=out["l_min"] / max(scene_scale, 1e-12) + 1e-6, max_val=1.6)
        * scene_scale
    )
    if out["l_max"] <= out["l_min"]:
        out["l_max"] = out["l_min"] + 1e-6
    return out


def _clip(value: float, lo: float, hi: float) -> float:
    return float(min(max(float(value), lo), hi))


def default_lengthscale_ratios(scene_scale: float) -> Dict[str, Any]:
    safe_scale = max(float(scene_scale), 1e-6)
    return {
        "tau_ratio": _clip(0.90 / safe_scale, 0.05, 2.50),
        "beta": 0.18,
        "l_min_ratio": 0.20,
        "l_max_ratio": 1.00,
        "radius_ratio": _clip(2.00 / safe_scale, 0.05, 4.00),
        "reasoning_summary": "Seeded from semantic defaults.",
    }


def sanitize_lengthscale_ratios(raw_value: Dict[str, Any] | None, *, fallback: Dict[str, Any]) -> Dict[str, Any]:
    value = dict(raw_value or {})
    out = {
        "tau_ratio": _clip(value.get("tau_ratio", fallback["tau_ratio"]), 0.05, 2.50),
        "beta": _clip(value.get("beta", fallback["beta"]), 0.0, 2.50),
        "l_min_ratio": _clip(value.get("l_min_ratio", fallback["l_min_ratio"]), 0.03, 1.00),
        "l_max_ratio": _clip(value.get("l_max_ratio", fallback["l_max_ratio"]), 0.04, 2.00),
        "radius_ratio": _clip(value.get("radius_ratio", fallback["radius_ratio"]), 0.05, 4.00),
        "reasoning_summary": str(value.get("reasoning_summary", fallback.get("reasoning_summary", "")))[:240],
    }
    if out["l_max_ratio"] <= out["l_min_ratio"]:
        out["l_max_ratio"] = min(2.00, out["l_min_ratio"] + 0.05)
    return out


def lengthscale_ratios_to_semantic_cfg(ratio_cfg: Dict[str, Any], *, scene_scale: float) -> Dict[str, Any]:
    ratio_cfg = sanitize_lengthscale_ratios(ratio_cfg, fallback=default_lengthscale_ratios(scene_scale))
    sem_cfg = sanitize_semantic_cfg(default_semantic_cfg())
    sem_cfg = build_world_scaled_semantic_cfg(sem_cfg, scene_scale=scene_scale)
    sem_cfg["tau"] = ratio_cfg["tau_ratio"] * scene_scale
    sem_cfg["beta"] = ratio_cfg["beta"]
    sem_cfg["l_min"] = ratio_cfg["l_min_ratio"] * scene_scale
    sem_cfg["l_max"] = ratio_cfg["l_max_ratio"] * scene_scale
    sem_cfg["radius"] = ratio_cfg["radius_ratio"] * scene_scale
    if sem_cfg["l_max"] <= sem_cfg["l_min"]:
        sem_cfg["l_max"] = sem_cfg["l_min"] + 1e-6
    sem_cfg["reasoning_summary"] = ratio_cfg.get("reasoning_summary", sem_cfg["reasoning_summary"])
    return sem_cfg


def semantic_cfg_to_lengthscale_ratios(sem_cfg: Dict[str, Any], *, scene_scale: float) -> Dict[str, Any]:
    safe_scale = max(float(scene_scale), 1e-6)
    fallback = default_lengthscale_ratios(scene_scale)
    return sanitize_lengthscale_ratios(
        {
            "tau_ratio": float(sem_cfg.get("tau", fallback["tau_ratio"])) / safe_scale,
            "beta": float(sem_cfg.get("beta", fallback["beta"])),
            "l_min_ratio": float(sem_cfg.get("l_min", fallback["l_min_ratio"])),
            "l_max_ratio": float(sem_cfg.get("l_max", fallback["l_max_ratio"])),
            "radius_ratio": float(sem_cfg.get("radius", fallback["radius_ratio"])),
            "reasoning_summary": str(sem_cfg.get("reasoning_summary", "LLM-derived semantic lengthscale.")),
        },
        fallback=fallback,
    )
