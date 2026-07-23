import numpy as np

from sk_gp.temporal import TemporalOutlierDetector


def test_semantic_info_gain_formula():
    # g_t = delta / (delta + 1), only positive deltas relax the threshold.
    assert TemporalOutlierDetector.semantic_info_gain(10, 10) == 0.0
    assert TemporalOutlierDetector.semantic_info_gain(11, 10) == 0.5
    assert TemporalOutlierDetector.semantic_info_gain(13, 10) == 3 / 4
    assert TemporalOutlierDetector.semantic_info_gain(5, 10) == 0.0  # clipped at 0


def test_detector_flags_off_manifold_jump():
    # Verified parameter fields drift smoothly along a low-dimensional manifold;
    # PCA captures that direction and flags deviations off it.
    det = TemporalOutlierDetector(window=6, tau0=9.0, n_components=2)
    rng = np.random.default_rng(0)
    d = 16
    base = rng.normal(0.0, 1.0, size=d)
    direction = rng.normal(0.0, 1.0, size=d)

    def on_manifold(t):
        return base + t * direction + rng.normal(0.0, 0.01, size=d)

    for t in np.linspace(0.0, 1.0, 6):
        det.update(on_manifold(t), n_entities=8)

    # A point on the manifold (within the observed range) is accepted.
    cont = det.update(on_manifold(0.5), n_entities=8)
    assert not cont.is_outlier

    # A large jump orthogonal to the manifold is flagged and substituted.
    orthogonal = rng.normal(0.0, 1.0, size=d)
    jump = base + 50.0 * orthogonal
    decision = det.update(jump, n_entities=8)
    assert decision.is_outlier
    assert np.allclose(decision.accepted, det._last_valid)  # substituted, not the jump


def test_info_gain_raises_threshold():
    det = TemporalOutlierDetector(window=5, tau0=5.0, n_components=2)
    rng = np.random.default_rng(1)
    base = rng.normal(0.0, 1.0, size=10)
    for _ in range(5):
        det.update(base + rng.normal(0.0, 0.05, size=10), n_entities=8)
    # With many newly discovered entities, the threshold is relaxed (info gain -> ~1).
    d = det.update(base + rng.normal(0.0, 0.05, size=10), n_entities=40)
    assert d.threshold > det.tau0
