"""Scene-graph environment container and JSON loader.

An :class:`Environment` is the parsed, geometry-ready form of a metric-semantic
scene graph (``.json``): rooms, walls (with materials), rectangular objects (with
materials), access points, spatial bounds and a material -> attenuation table.
This is the sole input the Semantic Kernel needs -- no material ground truth or AP
locations are required for the SK model itself (they are only used by the
simulator oracle to synthesize RSSI ground truth).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import LineString, box

__all__ = [
    "Environment",
    "DEFAULT_MATERIAL_ATTENUATION",
    "load_environment",
    "environment_from_scene",
    "compute_grid",
    "make_linestring",
]

# Material -> per-crossing attenuation (dB). Matches the simulator oracle.
DEFAULT_MATERIAL_ATTENUATION: Dict[str, float] = {
    "drywall": 3.0,
    "brick": 8.0,
    "tile": 5.0,
    "fabric": 1.0,
    "plastic": 2.0,
    "metal": 10.0,
    "wood": 4.0,
}


@dataclass
class Environment:
    rooms: dict
    walls: List[dict]          # {"line": LineString, "material": str}
    objects: List[dict]        # {"box": shapely.box, "material": str, "name": str}
    ap_list: List[np.ndarray]  # [np.array([x, y]), ...]
    bounds: Tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax)
    material_attenuation: Dict[str, float]


def environment_from_scene(
    data: dict,
    bounds_override: Optional[Tuple[float, float, float, float]] = None,
) -> Environment:
    """Build an :class:`Environment` from an in-memory scene-graph dict."""
    ap_list: List[np.ndarray] = []
    if isinstance(data.get("access_points"), list):
        for ap in data["access_points"]:
            ap_list.append(np.array(ap["pos"], dtype=np.float64))
    elif "access_point" in data:
        ap_list.append(np.array(data["access_point"]["pos"], dtype=np.float64))

    material_attenuation = dict(DEFAULT_MATERIAL_ATTENUATION)
    if "material_attenuation" in data:
        material_attenuation.update({k: float(v) for k, v in data["material_attenuation"].items()})

    rooms = data["rooms"]

    walls: List[dict] = []
    for room in rooms.values():
        for wall in room.get("walls", []):
            walls.append(
                {
                    "line": LineString(wall["segment"]),
                    "material": wall.get("material", "drywall"),
                }
            )

    objects: List[dict] = []
    for obj in data.get("objects", []):
        x, y = obj["pos"]
        w, h = obj["size"]
        objects.append(
            {
                "box": box(x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0),
                "material": obj.get("material", "wood"),
                "name": obj.get("name", ""),
            }
        )

    if bounds_override is not None:
        bounds = bounds_override
    elif "bounds" in data:
        b = data["bounds"]
        bounds = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    else:
        bounds = _infer_bounds_from_geometry(rooms, objects, ap_list)

    return Environment(
        rooms=rooms,
        walls=walls,
        objects=objects,
        ap_list=ap_list,
        bounds=bounds,
        material_attenuation=material_attenuation,
    )


def load_environment(
    json_path: str,
    bounds_override: Optional[Tuple[float, float, float, float]] = None,
) -> Environment:
    """Load an :class:`Environment` from a scene-graph JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return environment_from_scene(data, bounds_override=bounds_override)


def _infer_bounds_from_geometry(rooms, objects, ap_list, pad: float = 0.2) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    for room in rooms.values():
        for w in room.get("walls", []):
            for pt in w.get("segment", []):
                if len(pt) == 2:
                    xs.append(float(pt[0]))
                    ys.append(float(pt[1]))
        for pt in room.get("corners", []):
            if len(pt) == 2:
                xs.append(float(pt[0]))
                ys.append(float(pt[1]))
    for o in objects:
        x0, y0, x1, y1 = o["box"].bounds
        xs += [x0, x1]
        ys += [y0, y1]
    for ap in ap_list:
        xs.append(float(ap[0]))
        ys.append(float(ap[1]))

    if not xs or not ys:
        return (0.0, 9.0, 0.0, 8.0)
    return (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)


def compute_grid(
    env: Environment,
    nx: int = 300,
    ny: int = 300,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
):
    """Regular query grid over the environment bounds.

    Returns ``(X, Y, positions)`` where ``X, Y`` are ``(ny, nx)`` meshgrids and
    ``positions`` is the ``(nx*ny, 2)`` stack of query locations.
    """
    xmin, xmax, ymin, ymax = env.bounds
    if xlim is None:
        xlim = (xmin, xmax)
    if ylim is None:
        ylim = (ymin, ymax)
    x = np.linspace(float(xlim[0]), float(xlim[1]), nx, dtype=np.float64)
    y = np.linspace(float(ylim[0]), float(ylim[1]), ny, dtype=np.float64)
    X, Y = np.meshgrid(x, y)
    positions = np.column_stack((X.ravel(), Y.ravel())).astype(np.float64)
    return X, Y, positions


def make_linestring(a, b) -> LineString:
    return LineString([(float(a[0]), float(a[1])), (float(b[0]), float(b[1]))])
