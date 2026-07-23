"""Metric-semantic scene-graph data structures.

A :class:`SceneGraph` bundles rooms (each with materialed walls), rectangular
objects, doors (gaps in walls) and an access point.  ``to_json`` serializes it in
the same ``.json`` schema the Semantic Kernel consumes, and can optionally hide
materials and the AP position to emulate a partially observed graph (what the
robot actually perceives) versus the full oracle graph.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "Wall",
    "Room",
    "Obj",
    "Door",
    "AccessPoint",
    "SceneGraph",
    "MATERIALS_WALL",
    "MATERIALS_OBJ",
    "MATERIAL_ATTEN",
]

# Wall / object materials the generator draws from, and their attenuation (dB).
MATERIALS_WALL = ["drywall", "brick", "tile"]
MATERIALS_OBJ = ["fabric", "plastic", "metal", "wood"]

MATERIAL_ATTEN: Dict[str, float] = {
    "drywall": 3.0,
    "brick": 8.0,
    "tile": 5.0,
    "fabric": 1.0,
    "plastic": 2.0,
    "metal": 10.0,
    "wood": 4.0,
}


@dataclass
class Wall:
    segment: Tuple[Tuple[float, float], Tuple[float, float]]
    material: str


@dataclass
class Room:
    name: str
    corners: List[Tuple[float, float]]
    walls: List[Wall]


@dataclass
class Obj:
    name: str
    pos: Tuple[float, float]
    size: Tuple[float, float]
    material: str
    cls: str = "furniture"


@dataclass
class Door:
    """A door is a gap in a wall along ``segment``; ``rooms`` are the two sides
    (the second is ``None`` when the door leads outside)."""

    segment: Tuple[Tuple[float, float], Tuple[float, float]]
    rooms: Tuple[str, Optional[str]]


@dataclass
class AccessPoint:
    pos: Tuple[float, float]


@dataclass
class SceneGraph:
    rooms: Dict[str, Room]
    objects: List[Obj]
    access_point: AccessPoint
    doors: List[Door]

    def to_json(self, hide_materials: bool = False, hide_ap: Optional[bool] = None) -> Dict[str, Any]:
        if hide_ap is None:
            hide_ap = hide_materials

        rooms_json: Dict[str, Any] = {
            key: {
                "corners": value.corners,
                "walls": [
                    {"segment": wall.segment}
                    if hide_materials
                    else {"segment": wall.segment, "material": wall.material}
                    for wall in value.walls
                ],
            }
            for key, value in self.rooms.items()
        }

        objects_json: List[Dict[str, Any]] = [
            (
                {"name": obj.name, "pos": obj.pos, "size": obj.size, "cls": obj.cls}
                if hide_materials
                else asdict(obj)
            )
            for obj in self.objects
        ]

        doors_json: List[Dict[str, Any]] = [
            {"segment": door.segment, "rooms": [room for room in door.rooms if room is not None]}
            for door in self.doors
        ]

        data: Dict[str, Any] = {"rooms": rooms_json, "objects": objects_json, "doors": doors_json}
        if not hide_ap:
            data["access_point"] = {"pos": self.access_point.pos}
        return data
