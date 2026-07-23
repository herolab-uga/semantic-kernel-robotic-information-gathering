import numpy as np

from sk_gp.kernel.fields import LengthscaleField, PrimitiveField, primitive_weights


def test_primitive_weights_sum_to_one():
    x = np.random.default_rng(0).uniform(0, 5, size=(50, 2))
    centers = np.array([[1.0, 1.0], [4.0, 4.0]])
    w = primitive_weights(x, centers, tau=[1.0, 1.0], radius=[10.0, 10.0], theta=[1.0, 1.0])
    assert w.shape == (50, 2)
    assert np.allclose(w.sum(axis=1), 1.0)


def test_primitive_weights_hard_cutoff_falls_back_to_uniform():
    # A query far outside every influence radius -> uniform fallback (still sums to 1).
    x = np.array([[100.0, 100.0]])
    centers = np.array([[0.0, 0.0], [1.0, 1.0]])
    w = primitive_weights(x, centers, tau=[0.5, 0.5], radius=[1.0, 1.0], theta=[1.0, 1.0])
    assert np.allclose(w.sum(axis=1), 1.0)
    assert np.allclose(w, 0.5)


def test_primitive_field_blends_within_bounds():
    field = PrimitiveField(
        centers=[[0.0, 0.0], [5.0, 5.0]], values=[0.3, 1.4],
        tau=1.0, radius=10.0, v_min=0.3, v_max=1.4,
    )
    vals = field(np.array([[0.0, 0.0], [5.0, 5.0], [2.5, 2.5]]))
    assert vals.min() >= 0.3 - 1e-9
    assert vals.max() <= 1.4 + 1e-9
    # Nearest-primitive locations approach the corresponding primitive value.
    assert vals[0] < vals[1]


def test_lengthscale_field_clips_to_bounds():
    lf = LengthscaleField([[0, 0], [3, 3]], [0.4, 1.2], tau=1.0, radius=5.0, l_min=0.4, l_max=1.2)
    vals = lf(np.random.default_rng(1).uniform(0, 3, size=(30, 2)))
    assert vals.min() >= 0.4 - 1e-9 and vals.max() <= 1.2 + 1e-9
