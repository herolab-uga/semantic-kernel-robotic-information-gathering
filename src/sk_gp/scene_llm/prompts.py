"""Scene-LLM prompt templates (few-shot expert prompting).

A single, uniform prompt template is used during both fine-tuning and inference
(Section III).  It consists of:

* a concise **role description** (``LENGTHSCALE_SYSTEM_PROMPT`` /
  ``SEMANTIC_CFG_SYSTEM_PROMPT``),
* a **task contract** describing the available scene graph (``.json``),
* a **structured output schema** (JSON) compatible with the downstream
  deterministic verifier / kernel builders,
* **few-shot expert exemplars** with knowledge notes -- valid outputs that yield
  accurate reconstructions plus counter-examples that discourage common failure
  modes.

``build_scene_prompt`` renders the compact scene summary the model is conditioned
on; ``FEW_SHOT_KNOWLEDGE_NOTES`` documents the exemplar structure (populate with
your curated exemplars, e.g. 20 valid + 5 counter-examples as in the paper).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np

from .exemplars import KNOWLEDGE_NOTES as FEW_SHOT_KNOWLEDGE_NOTES
from .exemplars import render_few_shot_block

__all__ = [
    "LENGTHSCALE_SYSTEM_PROMPT",
    "SEMANTIC_CFG_SYSTEM_PROMPT",
    "FEW_SHOT_KNOWLEDGE_NOTES",
    "build_scene_prompt",
    "build_semantic_cfg_prompt",
]

# Role + task contract for the fine-tuned Scene-LLM (lengthscale ratios only).
LENGTHSCALE_SYSTEM_PROMPT = (
    "You tune only semantic lengthscale parameters for a gated nonstationary Gaussian "
    "process that predicts indoor RSSI maps. Return only one JSON object with keys "
    "tau_ratio, beta, l_min_ratio, l_max_ratio, radius_ratio, reasoning_summary. "
    "All *_ratio values are fractions of scene_scale. Keep l_max_ratio > l_min_ratio. "
    "Use shorter scales for denser geometry and more obstacles."
)

# Role + task contract for the full semantic-kernel config (few-shot, no fine-tuning).
SEMANTIC_CFG_SYSTEM_PROMPT = (
    "You are designing hyperparameters for an indoor RSSI Gaussian process. "
    "The downstream model is a semantic gated Gibbs kernel with a smooth geometry-aware gate. "
    "You must infer how sharply the RSSI field should change near walls, doors, "
    "room boundaries, and attenuating objects.\n"
    "Return only one JSON object. No markdown, no prose outside JSON.\n"
    "Choose values that make the field smoother in open rooms and shorter-scale "
    "near dense obstacles or high-attenuation materials.\n"
    "Use only these keys: K, priors, tau, beta, l_min, l_max, radius, temperature, "
    "center_mode, sem_variance, sem_jitter, sem_alpha, gate_material_params, "
    "normalize_y, reasoning_summary."
)

# The curated 20 valid + 5 counter-example exemplars live in `exemplars.py`;
# `FEW_SHOT_KNOWLEDGE_NOTES` re-exports their knowledge notes for convenience.


def _scene_summary(scene_json: Dict[str, Any], scene_scale: float) -> Dict[str, Any]:
    room_names = sorted(scene_json["rooms"].keys())
    room_areas: List[float] = []
    for room in scene_json["rooms"].values():
        (x0, y0), _, (x2, y2), _ = room["corners"]
        room_areas.append(abs((x2 - x0) * (y2 - y0)))
    materials = sorted(
        {str(w["material"]).strip().lower() for r in scene_json["rooms"].values() for w in r.get("walls", []) if "material" in w}.union(
            {str(o["material"]).strip().lower() for o in scene_json.get("objects", []) if "material" in o}
        )
    )
    return {
        "scene_scale": round(float(scene_scale), 6),
        "num_rooms": len(room_names),
        "num_objects": len(scene_json.get("objects", [])),
        "num_doors": len(scene_json.get("doors", [])),
        "room_names": room_names,
        "room_area_min": round(float(min(room_areas) if room_areas else 0.0), 6),
        "room_area_mean": round(float(np.mean(room_areas) if room_areas else 0.0), 6),
        "room_area_max": round(float(max(room_areas) if room_areas else 0.0), 6),
        "materials": materials,
    }


def build_scene_prompt(scene_json: Dict[str, Any], *, complexity: str, scene_scale: float) -> str:
    """Render the fine-tuning / inference prompt for the lengthscale-ratio task."""
    summary = {"complexity": complexity, **_scene_summary(scene_json, scene_scale)}
    constraints = {
        "tau_ratio_range": [0.05, 2.50],
        "beta_range": [0.0, 2.50],
        "l_min_ratio_range": [0.03, 1.00],
        "l_max_ratio_range": [0.04, 2.00],
        "radius_ratio_range": [0.05, 4.00],
        "note": "All *_ratio values are fractions of scene_scale.",
    }
    return (
        "Tune only the lengthscale-related parameters for the semantic GP.\n"
        "Return JSON only with keys: tau_ratio, beta, l_min_ratio, l_max_ratio, radius_ratio, reasoning_summary.\n"
        f"Constraints: {json.dumps(constraints, separators=(',', ':'))}\n\n"
        f"{render_few_shot_block()}\n\n"
        "Now do the same for this scene.\n"
        f"Scene summary: {json.dumps(summary, separators=(',', ':'))}\n"
        f"Scene JSON: {json.dumps(scene_json, separators=(',', ':'))}"
    )


def build_semantic_cfg_prompt(scene_json: Dict[str, Any], *, scene_scale: float, gate_distance_scale: float, materials: List[str]) -> str:
    """Render the few-shot prompt for the full semantic-kernel config.

    Reuses the curated lengthscale exemplars as expert guidance, then asks for the
    fuller config (adding gate/variance keys) for the specific scene.
    """
    materials_csv = ", ".join(materials)
    return (
        "Infer a semantic GP configuration for this indoor scene JSON.\n"
        f"l_min and l_max are FRACTIONS of scene_scale={scene_scale:.4f} m.\n"
        f"gate_material_params keyed by materials from [{materials_csv}], each {{lambda>=0, gamma>0}}; "
        f"gamma acts on distances normalized by gate_distance_scale={gate_distance_scale:.4f} m.\n\n"
        f"{render_few_shot_block()}\n\n"
        "Now produce the full semantic config for this scene.\n"
        f"Scene JSON: {json.dumps(scene_json, separators=(',', ':'))}"
    )
