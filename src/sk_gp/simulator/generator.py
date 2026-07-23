"""Procedural indoor scene generator.

``generate_scene`` lays out a grid of rooms, randomly merges a few of them,
assigns wall/object materials, carves doors (as missing walls) so every room is
reachable, scatters furniture, and drops a single access point.  ``complexity``
in ``{"simple", "medium", "complex"}`` controls the room-grid size -- the paper
trains the Scene-LLM on 200 such procedurally generated worlds (Section IV).
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np

from .scene_graph import (
    MATERIALS_OBJ,
    MATERIALS_WALL,
    AccessPoint,
    Door,
    Obj,
    Room,
    SceneGraph,
    Wall,
)

__all__ = ["generate_scene"]


def _rect_corners(x0: float, y0: float, x1: float, y1: float) -> List[Tuple[float, float]]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _norm_segment(p0, p1):
    return (p0, p1) if p0 <= p1 else (p1, p0)


def _room_from_rect(name, x0, y0, x1, y1, mat_a, mat_b) -> Room:
    corners = _rect_corners(x0, y0, x1, y1)
    walls = [
        Wall(((x0, y0), (x1, y0)), mat_a),
        Wall(((x1, y0), (x1, y1)), mat_b),
        Wall(((x1, y1), (x0, y1)), mat_a),
        Wall(((x0, y1), (x0, y0)), mat_b),
    ]
    return Room(name=name, corners=corners, walls=walls)


def generate_scene(seed: int = 42, complexity: str = "simple") -> SceneGraph:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    if complexity == "simple":
        nx, ny = 2, 2
    elif complexity in ("medium", "moderate"):
        nx, ny = 3, 2
    elif complexity == "complex":
        nx, ny = 4, 3
    else:
        raise ValueError("complexity must be simple|medium|complex (moderate aliases medium)")

    cell_w = rng.uniform(2.0, 3.0)
    cell_h = rng.uniform(2.0, 3.0)

    rects: List[Tuple[float, float, float, float]] = []
    for row in range(ny):
        for col in range(nx):
            x0, y0 = col * cell_w, row * cell_h
            rects.append((x0, y0, x0 + cell_w, y0 + cell_h))

    rooms_rects = rects[:]
    merges = 0
    max_merges = {"simple": 1, "medium": 2, "moderate": 2, "complex": 3}[complexity]
    while merges < max_merges and len(rooms_rects) > 1:
        a = rng.randrange(len(rooms_rects))
        ax0, ay0, ax1, ay1 = rooms_rects[a]
        candidates: List[int] = []
        for b, (bx0, by0, bx1, by1) in enumerate(rooms_rects):
            if b == a:
                continue
            if (abs(ax1 - bx0) < 1e-6 or abs(bx1 - ax0) < 1e-6) and not (ay1 <= by0 or by1 <= ay0):
                candidates.append(b)
            if (abs(ay1 - by0) < 1e-6 or abs(by1 - ay0) < 1e-6) and not (ax1 <= bx0 or bx1 <= ax0):
                candidates.append(b)
        if not candidates:
            break
        b = rng.choice(candidates)
        bx0, by0, bx1, by1 = rooms_rects[b]
        rooms_rects[a] = (min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1))
        del rooms_rects[b]
        merges += 1

    if complexity == "complex" and len(rooms_rects) < 8:
        while len(rooms_rects) < 8:
            idx = max(
                range(len(rooms_rects)),
                key=lambda k: (rooms_rects[k][2] - rooms_rects[k][0]) * (rooms_rects[k][3] - rooms_rects[k][1]),
            )
            x0, y0, x1, y1 = rooms_rects[idx]
            if (x1 - x0) > (y1 - y0):
                mid = 0.5 * (x0 + x1)
                rooms_rects[idx] = (x0, y0, mid, y1)
                rooms_rects.append((mid, y0, x1, y1))
            else:
                mid = 0.5 * (y0 + y1)
                rooms_rects[idx] = (x0, y0, x1, mid)
                rooms_rects.append((x0, mid, x1, y1))

    rooms: Dict[str, Room] = {}
    for idx, (x0, y0, x1, y1) in enumerate(rooms_rects):
        name = f"Room{idx + 1}"
        rooms[name] = _room_from_rect(name, x0, y0, x1, y1, rng.choice(MATERIALS_WALL), rng.choice(MATERIALS_WALL))

    edge_index: Dict = {}
    room_edges: Dict[str, List] = {}
    for room_name, room in rooms.items():
        room_edges[room_name] = []
        for wall_idx, wall in enumerate(room.walls):
            key = _norm_segment(*wall.segment)
            room_edges[room_name].append(key)
            edge_index.setdefault(key, []).append((room_name, wall_idx))

    adjacency_by_room: Dict[str, List] = {}
    for key, entries in edge_index.items():
        if len(entries) >= 2:
            for room_name, _ in entries:
                for other_name, _ in entries:
                    if other_name != room_name:
                        adjacency_by_room.setdefault(room_name, []).append((key[0], key[1], other_name))

    doors: List[Door] = []
    used_edges: set = set()
    has_door: Dict[str, bool] = {room_name: False for room_name in rooms}
    for room_name in rooms:
        if has_door[room_name]:
            continue
        interior = [
            (p0, p1, other)
            for (p0, p1, other) in adjacency_by_room.get(room_name, [])
            if _norm_segment(p0, p1) not in used_edges
        ]
        if interior:
            p0, p1, other = rng.choice(interior)
            seg_key = _norm_segment(p0, p1)
            doors.append(Door(segment=seg_key, rooms=(room_name, other)))
            used_edges.add(seg_key)
            has_door[room_name] = True
            has_door[other] = True
        else:
            ext = [e for e in room_edges.get(room_name, []) if e not in used_edges and len(edge_index.get(e, [])) == 1]
            if not ext:
                ext = [e for e in room_edges.get(room_name, []) if e not in used_edges]
            if ext:
                seg_key = rng.choice(ext)
                doors.append(Door(segment=seg_key, rooms=(room_name, None)))
                used_edges.add(seg_key)
                has_door[room_name] = True

    door_edges_by_room: Dict[str, set] = {room_name: set() for room_name in rooms}
    for door in doors:
        seg_key = _norm_segment(*door.segment)
        room_a, room_b = door.rooms
        if room_a in door_edges_by_room:
            door_edges_by_room[room_a].add(seg_key)
        if room_b is not None and room_b in door_edges_by_room:
            door_edges_by_room[room_b].add(seg_key)

    for room_name, room in rooms.items():
        room.walls = [w for w in room.walls if _norm_segment(*w.segment) not in door_edges_by_room[room_name]]

    objects: List[Obj] = []
    for room_name, room in rooms.items():
        num_obj = int(np_rng.integers(1, 4))
        x0, y0 = room.corners[0]
        x1, y1 = room.corners[2]
        for idx in range(num_obj):
            width = float(np_rng.uniform(0.3, 1.0))
            height = float(np_rng.uniform(0.3, 1.2))
            ox = float(np_rng.uniform(x0 + width / 2.0, x1 - width / 2.0))
            oy = float(np_rng.uniform(y0 + height / 2.0, y1 - height / 2.0))
            objects.append(
                Obj(name=f"Obj_{room_name}_{idx}", pos=(ox, oy), size=(width, height), material=rng.choice(MATERIALS_OBJ))
            )

    room_name = rng.choice(list(rooms))
    rx0, ry0 = rooms[room_name].corners[0]
    rx1, ry1 = rooms[room_name].corners[2]
    access_point = AccessPoint(
        pos=(float(np_rng.uniform(rx0 + 0.2, rx1 - 0.2)), float(np_rng.uniform(ry0 + 0.2, ry1 - 0.2)))
    )

    return SceneGraph(rooms=rooms, objects=objects, access_point=access_point, doors=doors)
