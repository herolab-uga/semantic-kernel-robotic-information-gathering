"""Radio-world oracle: semantics-aware RSSI ground truth and measurements.

Implements the simulator side of Eq. (12): a log-distance path-loss field with
per-crossing material attenuation, plus optional large-scale shadowing and
small-scale fading, plus i.i.d. measurement noise at two evaluation levels
(N1 = 5 dB, N2 = 10 dB).  A :class:`RadioWorld` bundles a :class:`SceneGraph`
with its rasterized true field and a GP surrogate, and is the unit consumed by
the fine-tuning loop (``world_stream`` / ``make_worlds``).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .scene_graph import MATERIAL_ATTEN, SceneGraph

__all__ = [
    "SIGMA_LARGE_DB",
    "SIGMA_SMALL_DB",
    "NOISE_LEVELS_DB",
    "log_distance_pathloss",
    "rssi_at_point",
    "scene_bounds",
    "rasterize_map",
    "sample_environment",
    "add_fading_noise",
    "RadioWorld",
    "make_worlds",
    "world_stream",
]

# Fading / noise parameters (Eq. 12 and Section IV).
SIGMA_LARGE_DB = 6.0   # large-scale shadowing std (correlated Gaussian field)
SIGMA_SMALL_DB = 1.0   # small-scale fading std
NOISE_LEVELS_DB = {"N1": 5.0, "N2": 10.0}  # measurement noise levels


def _segment_intersect(a1, a2, b1, b2) -> bool:
    def ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])

    return (ccw(a1, b1, b2) != ccw(a2, b1, b2)) and (ccw(a1, a2, b1) != ccw(a1, a2, b2))


def _line_intersects_rect(p1, p2, cx, cy, width, height) -> bool:
    x0, x1 = cx - width / 2.0, cx + width / 2.0
    y0, y1 = cy - height / 2.0, cy + height / 2.0
    edges = [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
    for e1, e2 in edges:
        if _segment_intersect(p1, p2, e1, e2):
            return True
    return (x0 <= p1[0] <= x1 and y0 <= p1[1] <= y1) or (x0 <= p2[0] <= x1 and y0 <= p2[1] <= y1)


def _atten_for_material(material: Optional[str]) -> float:
    if material is None:
        return 4.0
    return float(MATERIAL_ATTEN.get(material, 3.0))


def log_distance_pathloss(d: float, n: float = 3.0, p_tx_dbm: float = -30.0, d0: float = 1.0) -> float:
    d = max(float(d), 1e-2)
    return float(p_tx_dbm - 10.0 * n * math.log10(d / d0))


def rssi_at_point(scene_json: Dict[str, Any], x: float, y: float, n: float = 3.0, p_tx_dbm: float = -30.0) -> float:
    """Deterministic RSSI at ``(x, y)`` from the single AP (Eq. 12 core)."""
    apx, apy = scene_json["access_point"]["pos"]
    base = log_distance_pathloss(math.hypot(x - apx, y - apy), n=n, p_tx_dbm=p_tx_dbm)
    atten = 0.0
    for room in scene_json["rooms"].values():
        for wall in room["walls"]:
            seg = wall["segment"] if isinstance(wall, dict) else wall.segment
            material = wall.get("material") if isinstance(wall, dict) else getattr(wall, "material", None)
            if _segment_intersect((apx, apy), (x, y), seg[0], seg[1]):
                atten += _atten_for_material(material)
    for obj in scene_json["objects"]:
        cx, cy = obj.get("pos", (0.0, 0.0))
        width, height = obj.get("size", (0.5, 0.5))
        if _line_intersects_rect((apx, apy), (x, y), cx, cy, width, height):
            atten += 0.5 * _atten_for_material(obj.get("material"))
    return float(base - atten)


def scene_bounds(scene_json: Dict[str, Any]) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    for room in scene_json["rooms"].values():
        for x, y in room["corners"]:
            xs.append(float(x))
            ys.append(float(y))
    return min(xs) - 0.2, min(ys) - 0.2, max(xs) + 0.2, max(ys) + 0.2


def rasterize_map(scene_json: Dict[str, Any], res: int = 160, n: float = 3.0, p_tx_dbm: float = -30.0):
    x0, y0, x1, y1 = scene_bounds(scene_json)
    xs = np.linspace(x0, x1, res)
    ys = np.linspace(y0, y1, res)
    grid = np.zeros((res, res), dtype=np.float32)
    for row_idx, yy in enumerate(ys):
        for col_idx, xx in enumerate(xs):
            grid[row_idx, col_idx] = rssi_at_point(scene_json, float(xx), float(yy), n=n, p_tx_dbm=p_tx_dbm)
    return xs, ys, grid


def add_fading_noise(
    values: np.ndarray,
    noise_level_db: float = 0.0,
    sigma_large_db: float = 0.0,
    sigma_small_db: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Add large-/small-scale fading and i.i.d. measurement noise (Eq. 12)."""
    rng = np.random.default_rng() if rng is None else rng
    out = np.asarray(values, dtype=np.float64).copy()
    if sigma_large_db > 0.0:
        out += rng.normal(0.0, sigma_large_db)  # single correlated shadowing offset
    if sigma_small_db > 0.0:
        out += rng.normal(0.0, sigma_small_db, size=out.shape)
    if noise_level_db > 0.0:
        out += rng.normal(0.0, noise_level_db, size=out.shape)
    return out


def sample_environment(scene_json: Dict[str, Any], num_samples: int = 1000, n: float = 3.0, p_tx_dbm: float = -30.0):
    x0, y0, x1, y1 = scene_bounds(scene_json)
    xs = np.random.uniform(x0, x1, size=(num_samples,))
    ys = np.random.uniform(y0, y1, size=(num_samples,))
    xy = np.stack([xs, ys], axis=1)
    rss = np.array([rssi_at_point(scene_json, float(x), float(y), n=n, p_tx_dbm=p_tx_dbm) for x, y in xy], dtype=np.float32)
    return xy, rss


def _gp_fit_and_mean_map(xy: np.ndarray, rss: np.ndarray, xs: np.ndarray, ys: np.ndarray):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    xx, yy = np.meshgrid(xs, ys)
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1)
    kernel = (
        ConstantKernel(1.0, (1e-2, 10.0)) * RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0))
        + WhiteKernel(noise_level=2.0, noise_level_bounds=(1e-3, 10.0))
    )
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
    gp.fit(xy, rss)
    mean, std = gp.predict(coords, return_std=True)
    return mean.reshape(len(ys), len(xs)).astype(np.float32), std.reshape(len(ys), len(xs)).astype(np.float32), gp


class RadioWorld:
    """A scene graph plus its true RSSI field and an RBF-GP surrogate."""

    def __init__(self, scene: SceneGraph, n: float = 3.0, p_tx_dbm: float = -30.0, res: int = 160, fit_surrogate: bool = True):
        self.scene = scene
        self.scene_json = scene.to_json(hide_materials=False, hide_ap=False)
        self.n = float(n)
        self.p_tx_dbm = float(p_tx_dbm)
        self.res = int(res)
        self.xs, self.ys, self.true_map = rasterize_map(self.scene_json, res=res, n=n, p_tx_dbm=p_tx_dbm)
        if fit_surrogate:
            self.samples_xy, self.samples_rss = sample_environment(self.scene_json, num_samples=1000, n=n, p_tx_dbm=p_tx_dbm)
            self.gp_mean, self.gp_std, self.gp = _gp_fit_and_mean_map(self.samples_xy, self.samples_rss, self.xs, self.ys)
        else:
            self.samples_xy = self.samples_rss = None
            self.gp_mean = self.gp_std = self.gp = None

    def scene_graph_pair(self) -> Dict[str, Dict[str, Any]]:
        """Return the oracle (full) and observed (materials/AP hidden) graphs."""
        return {
            "ground_truth": self.scene.to_json(hide_materials=False, hide_ap=False),
            "observed": self.scene.to_json(hide_materials=True, hide_ap=True),
        }

    def locate_source(self) -> Dict[str, Any]:
        idx = int(np.argmax(self.gp_mean))
        row_idx, col_idx = np.unravel_index(idx, self.gp_mean.shape)
        return {"estimate": (float(self.xs[col_idx]), float(self.ys[row_idx])), "method": "argmax_gp_mean"}


def make_worlds(n: int, seed0: int, res: int) -> List[RadioWorld]:
    from .generator import generate_scene

    worlds: List[RadioWorld] = []
    for idx in range(int(n)):
        complexity = "complex" if idx % 3 == 2 else ("medium" if idx % 3 == 1 else "simple")
        worlds.append(RadioWorld(generate_scene(seed=seed0 + idx, complexity=complexity), res=res))
    return worlds


def world_stream(n_simple: int, n_medium: int, n_complex: int, seed0: int, res: int) -> Iterable[Tuple[int, RadioWorld, str]]:
    """Yield ``(scene_idx, RadioWorld, complexity)`` while incrementing the seed."""
    from .generator import generate_scene

    scene_idx = 0
    cur_seed = int(seed0)
    for complexity, count in (("simple", n_simple), ("medium", n_medium), ("complex", n_complex)):
        for _ in range(int(count)):
            world = RadioWorld(generate_scene(seed=cur_seed, complexity=complexity), res=int(res))
            yield scene_idx, world, complexity
            scene_idx += 1
            cur_seed += 1
