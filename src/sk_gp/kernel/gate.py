"""Semantic gate ``G(x, x')`` (Eq. 3-4).

The gate provides a smooth, geometry- and semantics-aware mechanism that
regulates correlations across scene entities.  It has two multiplicative parts:

1. **Geometry-aware attenuation** ``A(x, x')`` (Eq. 3).  For every entity ``e``
   (wall or object) we precompute a signed distance field ``s_e(x)``.  Two points
   on the *same* side of ``e`` (``s_e(x) s_e(x') > 0``) are left almost
   uncorrelated-preserving; points on *opposite* sides are attenuated with a
   strength governed by the entity's material attenuation ``lambda_e`` and a
   sharpness ``gamma`` acting on distances normalized by ``gate_distance_scale``.

2. **Semantic co-membership** ``z(x)^T z(x')`` (Eq. 4).  Provided by
   :class:`~sk_gp.kernel.hyperfield.SemanticHyperField`.

The two combine as ``G(x, x') = (z(x)^T z(x')) * A(x, x')`` and, by construction,
``G(x, x') in [0, 1]`` -- bounded attenuation that keeps GP inference numerically
stable.  ``build_gate_entities`` converts an :class:`Environment` plus per-material
``{lambda, gamma}`` parameters (from the Scene-LLM) into the entity list consumed
by the kernel.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..envs.environment import Environment

__all__ = [
    "build_gate_entities",
    "resolve_gate_material_params",
    "signed_distance_to_wall",
    "signed_distance_to_box",
]


def signed_distance_to_wall(points: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Signed distance from ``points`` to a wall segment ``[p0, p1]``.

    The sign encodes which side of the (oriented) segment a point lies on, so
    ``s_e(x) s_e(x') > 0`` iff both points are on the same side.
    """
    direction = p1 - p0
    length_sq = float(np.dot(direction, direction))
    if length_sq <= 1e-12:
        return np.linalg.norm(points - p0[None, :], axis=1)

    rel = points - p0[None, :]
    t = np.clip((rel @ direction) / length_sq, 0.0, 1.0)
    proj = p0[None, :] + t[:, None] * direction[None, :]
    dist = np.linalg.norm(points - proj, axis=1)
    cross = direction[0] * rel[:, 1] - direction[1] * rel[:, 0]
    sign = np.where(cross >= 0.0, 1.0, -1.0)
    return sign * dist


def signed_distance_to_box(points: np.ndarray, bounds) -> np.ndarray:
    """Signed distance to an axis-aligned box (negative inside, positive outside)."""
    x0, y0, x1, y1 = bounds
    px = points[:, 0]
    py = points[:, 1]

    dx_out = np.maximum(np.maximum(x0 - px, 0.0), px - x1)
    dy_out = np.maximum(np.maximum(y0 - py, 0.0), py - y1)
    outside_dist = np.hypot(dx_out, dy_out)

    inside = (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)
    dist_to_edge = np.minimum.reduce([px - x0, x1 - px, py - y0, y1 - py])

    signed = outside_dist
    signed[inside] = -dist_to_edge[inside]
    return signed


def resolve_gate_material_params(
    env: Environment,
    llm_params: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Fill in ``{lambda, gamma}`` for every material in the scene.

    LLM-provided values take precedence; missing materials fall back to a monotone
    function of the material's attenuation, so higher-attenuation materials get a
    stronger gate by default.
    """
    materials = {str(w["material"]).strip().lower() for w in env.walls}
    materials.update(str(o["material"]).strip().lower() for o in env.objects)

    resolved: Dict[str, Dict[str, float]] = {}
    for material in sorted(materials):
        attenuation = float(env.material_attenuation.get(material, 3.0))
        fallback = {
            "lambda": float(np.clip(0.05 * attenuation, 0.05, 0.75)),
            "gamma": float(np.clip(0.8 + 0.18 * attenuation, 0.25, 4.0)),
        }
        raw = llm_params.get(material, {})
        resolved[material] = {
            "lambda": float(np.clip(float(raw.get("lambda", fallback["lambda"])), 0.0, 2.0)),
            "gamma": float(np.clip(float(raw.get("gamma", fallback["gamma"])), 0.05, 12.0)),
        }
    return resolved


def build_gate_entities(
    env: Environment,
    gate_material_params: Dict[str, Dict[str, float]],
) -> List[Dict[str, Any]]:
    """Convert walls/objects into the gate-entity list used by the kernel."""
    entities: List[Dict[str, Any]] = []

    for wall in env.walls:
        coords = np.asarray(wall["line"].coords, dtype=np.float64)
        if coords.shape[0] < 2:
            continue
        material = str(wall["material"]).strip().lower()
        params = gate_material_params.get(material, {"lambda": 0.0, "gamma": 1.0})
        entities.append(
            {
                "kind": "wall",
                "material": material,
                "p0": coords[0],
                "p1": coords[-1],
                "lambda": float(params["lambda"]),
                "gamma": float(params["gamma"]),
            }
        )

    for obj in env.objects:
        x0, y0, x1, y1 = obj["box"].bounds
        material = str(obj["material"]).strip().lower()
        params = gate_material_params.get(material, {"lambda": 0.0, "gamma": 1.0})
        entities.append(
            {
                "kind": "object",
                "material": material,
                "bounds": (float(x0), float(y0), float(x1), float(y1)),
                "lambda": float(params["lambda"]),
                "gamma": float(params["gamma"]),
            }
        )

    return entities
