"""Expert-curated few-shot exemplars for the Scene-LLM (Section III).

The paper conditions the Scene-LLM on a uniform prompt that includes
**20 valid exemplars** (scene → kernel parameterization that yields accurate
reconstructions) and **5 counter-examples** (common failure modes to discourage).
These are those exemplars, in the fine-tuned/inference *lengthscale-ratio* schema::

    {tau_ratio, beta, l_min_ratio, l_max_ratio, radius_ratio, reasoning_summary}

All ``*_ratio`` values are fractions of ``scene_scale`` (so they transfer across
environments of different physical size). Semantic intuition encoded here:

* **Open, low-clutter, low-attenuation** space → long lengthscale
  (high ``l_max_ratio``), weak semantic shortening (low ``beta``), wide entity
  influence (high ``radius_ratio``).
* **Dense clutter / high-attenuation materials** (metal, brick, tile) → short
  lengthscale (low ``l_min_ratio``/``l_max_ratio``), strong shortening (high
  ``beta``), tight influence (low ``radius_ratio``).

``render_few_shot_block`` formats them for inclusion in the prompt;
``iter_valid_outputs`` exposes the raw JSON outputs (e.g. for validation/tests).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

__all__ = [
    "VALID_EXEMPLARS",
    "COUNTER_EXEMPLARS",
    "KNOWLEDGE_NOTES",
    "render_few_shot_block",
    "iter_valid_outputs",
]


def _valid(scene: str, output: Dict[str, Any], note: str) -> Dict[str, Any]:
    return {"scene": scene, "output": output, "note": note}


# -- 20 valid exemplars ----------------------------------------------------
VALID_EXEMPLARS: List[Dict[str, Any]] = [
    _valid(
        "Single large open hall, drywall perimeter only, no interior obstacles; strong line-of-sight.",
        {"tau_ratio": 0.30, "beta": 0.08, "l_min_ratio": 0.30, "l_max_ratio": 1.25,
         "radius_ratio": 0.55, "reasoning_summary": "Open LOS hall: long, smooth field; weak semantic shortening."},
        "Open halls allow signals to propagate freely over distance -> long lengthscale, low beta.",
    ),
    _valid(
        "Small dense storage room packed with metal shelving; frequent metal crossings.",
        {"tau_ratio": 0.10, "beta": 1.30, "l_min_ratio": 0.05, "l_max_ratio": 0.35,
         "radius_ratio": 0.15, "reasoning_summary": "Metal clutter attenuates sharply -> short scale, strong gate."},
        "Metal is the highest-attenuation material -> shortest lengthscale, high beta, tight radius.",
    ),
    _valid(
        "Typical multi-room drywall apartment, light furniture, a few doors between rooms.",
        {"tau_ratio": 0.22, "beta": 0.35, "l_min_ratio": 0.18, "l_max_ratio": 0.85,
         "radius_ratio": 0.35, "reasoning_summary": "Moderate drywall partitioning -> balanced non-stationarity."},
        "Drywall gives moderate attenuation; doors preserve some cross-room correlation.",
    ),
    _valid(
        "Long narrow corridor, drywall walls, sparse objects; propagation channeled along its length.",
        {"tau_ratio": 0.18, "beta": 0.25, "l_min_ratio": 0.15, "l_max_ratio": 0.90,
         "radius_ratio": 0.30, "reasoning_summary": "Corridor channels signal: long along axis, walls confine sides."},
        "Corridors sustain longer correlation along the open axis than across walls.",
    ),
    _valid(
        "Brick-walled office with several partitions and a metal filing cabinet cluster.",
        {"tau_ratio": 0.14, "beta": 0.90, "l_min_ratio": 0.08, "l_max_ratio": 0.50,
         "radius_ratio": 0.22, "reasoning_summary": "Brick + metal clusters -> short scale, strong attenuation."},
        "Brick (~8 dB) and metal drive short lengthscales near partitions.",
    ),
    _valid(
        "Tiled bathroom, small, reflective surfaces; moderate multipath.",
        {"tau_ratio": 0.16, "beta": 0.55, "l_min_ratio": 0.12, "l_max_ratio": 0.55,
         "radius_ratio": 0.25, "reasoning_summary": "Tile reflects and scatters -> moderately short, noisier field."},
        "Tiled/wet rooms reflect and scatter -> moderate lengthscale, elevated local variability.",
    ),
    _valid(
        "Open-plan office: large area, low cubicle partitions (fabric), few walls.",
        {"tau_ratio": 0.28, "beta": 0.20, "l_min_ratio": 0.22, "l_max_ratio": 1.05,
         "radius_ratio": 0.45, "reasoning_summary": "Low fabric partitions barely attenuate -> mostly long scale."},
        "Fabric (~1 dB) barely attenuates; keep lengthscale long despite partitions.",
    ),
    _valid(
        "Two large rooms joined by a wide doorway, drywall, minimal furniture.",
        {"tau_ratio": 0.26, "beta": 0.18, "l_min_ratio": 0.20, "l_max_ratio": 1.00,
         "radius_ratio": 0.42, "reasoning_summary": "Wide doorway keeps rooms correlated -> long scale, mild gate."},
        "Doors are gaps in walls -> preserve cross-room correlation; lengthen scale near openings.",
    ),
    _valid(
        "Cluttered workshop: mixed wood benches and metal machines throughout.",
        {"tau_ratio": 0.12, "beta": 1.05, "l_min_ratio": 0.07, "l_max_ratio": 0.45,
         "radius_ratio": 0.20, "reasoning_summary": "Dense mixed clutter with metal -> short, strongly gated field."},
        "Object density plus metal -> aggressive shortening; small influence radius per object.",
    ),
    _valid(
        "Six-room house, drywall interior, brick exterior, normal furnishing.",
        {"tau_ratio": 0.20, "beta": 0.45, "l_min_ratio": 0.15, "l_max_ratio": 0.80,
         "radius_ratio": 0.32, "reasoning_summary": "Mixed drywall/brick house -> moderate heterogeneity."},
        "Interior drywall + exterior brick -> moderate lengthscale, room-scale heterogeneity.",
    ),
    _valid(
        "Nine-room school wing, long drywall corridors linking classrooms.",
        {"tau_ratio": 0.20, "beta": 0.40, "l_min_ratio": 0.16, "l_max_ratio": 0.88,
         "radius_ratio": 0.34, "reasoning_summary": "Large layout, corridors + classrooms -> moderate, corridor-biased."},
        "Larger layouts keep ratios stable (scale-relative); corridors extend correlation.",
    ),
    _valid(
        "Server room dense with metal racks, elevated interference near equipment.",
        {"tau_ratio": 0.09, "beta": 1.45, "l_min_ratio": 0.05, "l_max_ratio": 0.30,
         "radius_ratio": 0.14, "reasoning_summary": "Wall-to-wall metal racks -> shortest scale, strongest gate."},
        "Extreme metal density -> push toward the lower bounds of lengthscale.",
    ),
    _valid(
        "Bedroom with wooden furniture (bed, wardrobe, desk), drywall walls.",
        {"tau_ratio": 0.19, "beta": 0.40, "l_min_ratio": 0.16, "l_max_ratio": 0.78,
         "radius_ratio": 0.30, "reasoning_summary": "Wood furniture (~4 dB) -> moderate, object-localized shortening."},
        "Wood attenuates moderately; shorten locally around furniture, not globally.",
    ),
    _valid(
        "Conference room: one large table, chairs, drywall, single door.",
        {"tau_ratio": 0.24, "beta": 0.22, "l_min_ratio": 0.20, "l_max_ratio": 0.95,
         "radius_ratio": 0.40, "reasoning_summary": "Mostly open room, central furniture -> long with mild dip."},
        "A single central object barely perturbs an otherwise open room.",
    ),
    _valid(
        "Kitchen with metal appliances (fridge, oven) along one wall, tile floor.",
        {"tau_ratio": 0.13, "beta": 0.85, "l_min_ratio": 0.09, "l_max_ratio": 0.55,
         "radius_ratio": 0.24, "reasoning_summary": "Metal appliances on one side -> asymmetric short-scale gate."},
        "Localized metal appliances create strong one-sided attenuation.",
    ),
    _valid(
        "Warehouse: very large, sparse tall metal shelving in rows with wide aisles.",
        {"tau_ratio": 0.26, "beta": 0.55, "l_min_ratio": 0.12, "l_max_ratio": 0.85,
         "radius_ratio": 0.30, "reasoning_summary": "Wide aisles stay long; shelving rows gate across them."},
        "Large open aisles keep long scale; metal rows gate correlation across aisles.",
    ),
    _valid(
        "Studio apartment, single room, drywall, moderate furniture, one AP.",
        {"tau_ratio": 0.23, "beta": 0.30, "l_min_ratio": 0.18, "l_max_ratio": 0.92,
         "radius_ratio": 0.38, "reasoning_summary": "Single open-ish room -> fairly long, light furniture dips."},
        "Single-room scenes favor longer scales with small local reductions.",
    ),
    _valid(
        "Classroom with rows of wooden desks, drywall, whiteboard wall.",
        {"tau_ratio": 0.18, "beta": 0.42, "l_min_ratio": 0.15, "l_max_ratio": 0.80,
         "radius_ratio": 0.30, "reasoning_summary": "Regular wooden desk rows -> moderate, periodic shortening."},
        "Repeated wood furniture -> moderate, spatially periodic non-stationarity.",
    ),
    _valid(
        "Lab with mixed materials: brick partition, plastic benches, some metal instruments.",
        {"tau_ratio": 0.15, "beta": 0.70, "l_min_ratio": 0.10, "l_max_ratio": 0.62,
         "radius_ratio": 0.26, "reasoning_summary": "Mixed brick/metal/plastic -> moderately short, heterogeneous."},
        "Weight the gate by material: brick/metal dominate, plastic (~2 dB) contributes little.",
    ),
    _valid(
        "Atrium/lobby: very open, high ceiling, glass and drywall, almost no obstacles.",
        {"tau_ratio": 0.34, "beta": 0.06, "l_min_ratio": 0.32, "l_max_ratio": 1.35,
         "radius_ratio": 0.60, "reasoning_summary": "Near-empty atrium -> longest, smoothest field of all scenes."},
        "The most open scenes take the longest lengthscales and the widest influence radius.",
    ),
]


def _counter(scene: str, bad_output: Dict[str, Any], why_wrong: str, note: str) -> Dict[str, Any]:
    return {"scene": scene, "bad_output": bad_output, "why_wrong": why_wrong, "note": note}


# -- 5 counter-examples (failure modes to avoid) ---------------------------
COUNTER_EXEMPLARS: List[Dict[str, Any]] = [
    _counter(
        "Open drywall hall.",
        {"tau_ratio": 0.30, "beta": 0.10, "l_min_ratio": 0.90, "l_max_ratio": 0.40,
         "radius_ratio": 0.5, "reasoning_summary": "..."},
        "l_max_ratio (0.40) <= l_min_ratio (0.90): invalid ordering, would collapse the field.",
        "Always keep l_max_ratio strictly greater than l_min_ratio.",
    ),
    _counter(
        "Dense metal storage room.",
        {"tau_ratio": 0.22, "beta": 0.18, "l_min_ratio": 0.20, "l_max_ratio": 1.00,
         "radius_ratio": 0.40, "reasoning_summary": "Generic defaults."},
        "Returns open-space defaults for a metal-dense room: ignores semantics entirely.",
        "Do not emit generic defaults blindly; condition the values on materials and clutter.",
    ),
    _counter(
        "Lightly furnished drywall bedroom.",
        {"tau_ratio": 0.20, "beta": 2.50, "l_min_ratio": 0.15, "l_max_ratio": 0.80,
         "radius_ratio": 0.30, "reasoning_summary": "..."},
        "beta pinned at the maximum collapses almost everywhere to l_min: over-attenuation.",
        "Reserve very high beta for genuinely dense/high-attenuation scenes, not light furnishing.",
    ),
    _counter(
        "Office with a brick partition wall.",
        {"tau_ratio": 0.20, "beta": 0.80, "l_min_ratio": 0.10, "l_max_ratio": 0.60,
         "radius_ratio": 0.05, "reasoning_summary": "..."},
        "radius_ratio at the floor gives entities almost no influence, so the wall is ignored.",
        "Set radius_ratio wide enough that walls/objects actually shape the nearby field.",
    ),
    _counter(
        "Multi-room drywall house.",
        {"tau_ratio": 0.2, "beta": 0.4, "l_min_ratio": 0.15, "l_max_ratio": 0.85,
         "note": "here are the values", "confidence": "high"},
        "Extra keys and prose outside the schema break the downstream deterministic parser.",
        "Output must strictly conform to the schema keys only: tau_ratio, beta, l_min_ratio, l_max_ratio, radius_ratio, reasoning_summary.",
    ),
]

# Expert knowledge notes surfaced alongside the exemplars.
KNOWLEDGE_NOTES: List[str] = [
    "All *_ratio values are fractions of scene_scale so they transfer across environment sizes.",
    "Rank materials by attenuation: metal > brick > tile > wood > plastic > drywall > fabric.",
    "Open, low-clutter space -> long lengthscale, low beta, wide radius.",
    "Dense clutter / high-attenuation materials -> short lengthscale, high beta, tight radius.",
    "Doors are gaps in walls: preserve cross-room correlation near openings.",
    "If required information is missing, return status = 'need_more_info'.",
    "Output must strictly conform to the defined output JSON schema.",
]


def iter_valid_outputs() -> List[Dict[str, Any]]:
    """Return just the JSON outputs of the valid exemplars (for tests/validation)."""
    return [dict(ex["output"]) for ex in VALID_EXEMPLARS]


def render_few_shot_block(max_valid: int | None = None, max_counter: int | None = None) -> str:
    """Format the exemplars + knowledge notes as a prompt-ready text block."""
    valid = VALID_EXEMPLARS if max_valid is None else VALID_EXEMPLARS[:max_valid]
    counter = COUNTER_EXEMPLARS if max_counter is None else COUNTER_EXEMPLARS[:max_counter]

    lines: List[str] = ["Knowledge notes:"]
    lines += [f"- {n}" for n in KNOWLEDGE_NOTES]
    lines.append("")
    lines.append(f"Valid exemplars ({len(valid)}):")
    for i, ex in enumerate(valid, 1):
        lines.append(f"{i}. Scene: {ex['scene']}")
        lines.append(f"   Output: {json.dumps(ex['output'], separators=(',', ':'))}")
        lines.append(f"   Note: {ex['note']}")
    lines.append("")
    lines.append(f"Counter-examples ({len(counter)}) — do NOT do these:")
    for i, ex in enumerate(counter, 1):
        lines.append(f"{i}. Scene: {ex['scene']}")
        lines.append(f"   Bad output: {json.dumps(ex['bad_output'], separators=(',', ':'))}")
        lines.append(f"   Why wrong: {ex['why_wrong']}")
    return "\n".join(lines)
