"""Procedural indoor scene generator and semantics-aware radio-world oracle."""

from .generator import generate_scene
from .radio_world import (
    NOISE_LEVELS_DB,
    SIGMA_LARGE_DB,
    SIGMA_SMALL_DB,
    RadioWorld,
    add_fading_noise,
    log_distance_pathloss,
    make_worlds,
    rasterize_map,
    rssi_at_point,
    sample_environment,
    scene_bounds,
    world_stream,
)
from .scene_graph import (
    MATERIAL_ATTEN,
    MATERIALS_OBJ,
    MATERIALS_WALL,
    AccessPoint,
    Door,
    Obj,
    Room,
    SceneGraph,
    Wall,
)

__all__ = [
    "generate_scene",
    "Wall",
    "Room",
    "Obj",
    "Door",
    "AccessPoint",
    "SceneGraph",
    "MATERIALS_WALL",
    "MATERIALS_OBJ",
    "MATERIAL_ATTEN",
    "RadioWorld",
    "make_worlds",
    "world_stream",
    "rssi_at_point",
    "rasterize_map",
    "sample_environment",
    "scene_bounds",
    "log_distance_pathloss",
    "add_fading_noise",
    "SIGMA_LARGE_DB",
    "SIGMA_SMALL_DB",
    "NOISE_LEVELS_DB",
]
