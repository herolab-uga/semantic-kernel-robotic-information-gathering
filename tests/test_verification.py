from sk_gp.scene_llm import build_world_scaled_semantic_cfg, default_semantic_cfg, sanitize_semantic_cfg
from sk_gp.verification import AdmissibleBounds, FormalVerifier, verify_parameters


def test_default_config_is_admissible():
    cfg = build_world_scaled_semantic_cfg(sanitize_semantic_cfg(default_semantic_cfg()), scene_scale=7.0)
    result = verify_parameters(cfg, bounds=AdmissibleBounds(), domain=(0.0, 9.0, 0.0, 8.0))
    assert result.valid, result.failures


def test_lengthscale_ordering_violation_is_rejected():
    result = verify_parameters({"l_min": 1.0, "l_max": 0.5})
    assert not result.valid
    assert any("l_max" in f for f in result.failures)


def test_negative_variance_and_noise_rejected():
    result = verify_parameters({"variance": -1.0, "noise": -0.5})
    assert not result.valid
    assert result.failure_log()


def test_out_of_domain_center_rejected():
    result = verify_parameters({"centers": [[100.0, 100.0]]}, domain=(0.0, 9.0, 0.0, 8.0))
    assert not result.valid
    assert any("outside domain" in f for f in result.failures)


def test_formal_verifier_retry_bookkeeping():
    verifier = FormalVerifier(domain=(0.0, 9.0, 0.0, 8.0), retry_limit=2)
    params, result = verifier.review({"l_min": 2.0, "l_max": 1.0})
    assert params is None and not result.valid
    assert verifier.attempts == 1 and verifier.should_retry
    verifier.review({"l_min": 2.0, "l_max": 1.0})
    assert not verifier.should_retry  # retry limit reached
