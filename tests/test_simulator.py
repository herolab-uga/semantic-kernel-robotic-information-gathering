import numpy as np

from sk_gp.envs import environment_from_scene
from sk_gp.simulator import generate_scene, rssi_at_point, scene_bounds


def test_generate_scene_is_deterministic():
    a = generate_scene(seed=7, complexity="medium").to_json()
    b = generate_scene(seed=7, complexity="medium").to_json()
    assert a == b


def test_scene_has_expected_structure():
    scene = generate_scene(seed=1, complexity="complex")
    sj = scene.to_json(hide_materials=False, hide_ap=False)
    assert len(sj["rooms"]) >= 1
    assert "access_point" in sj
    # Every room keeps at least one wall after doors are carved.
    assert all("walls" in room for room in sj["rooms"].values())


def test_rssi_decreases_with_distance():
    scene = generate_scene(seed=5, complexity="simple")
    sj = scene.to_json(hide_materials=False, hide_ap=False)
    apx, apy = sj["access_point"]["pos"]
    near = rssi_at_point(sj, apx + 0.3, apy + 0.3)
    far = rssi_at_point(sj, apx + 3.0, apy + 3.0)
    assert near > far


def test_observed_graph_hides_materials_and_ap():
    scene = generate_scene(seed=9, complexity="simple")
    observed = scene.to_json(hide_materials=True, hide_ap=True)
    assert "access_point" not in observed
    for room in observed["rooms"].values():
        for wall in room["walls"]:
            assert "material" not in wall


def test_environment_from_generated_scene_loads():
    scene = generate_scene(seed=3, complexity="medium")
    env = environment_from_scene(scene.to_json(hide_materials=False, hide_ap=False))
    assert len(env.walls) > 0
    xmin, xmax, ymin, ymax = env.bounds
    assert xmax > xmin and ymax > ymin
