from sk_gp.scene_llm import COUNTER_EXEMPLARS, VALID_EXEMPLARS, render_few_shot_block
from sk_gp.scene_llm.config import default_lengthscale_ratios, sanitize_lengthscale_ratios

RATIO_KEYS = {"tau_ratio", "beta", "l_min_ratio", "l_max_ratio", "radius_ratio", "reasoning_summary"}


def test_counts_match_paper():
    # Paper: 20 valid exemplars + 5 counter-examples.
    assert len(VALID_EXEMPLARS) == 20
    assert len(COUNTER_EXEMPLARS) == 5


def test_valid_exemplars_conform_to_schema_and_ranges():
    fallback = default_lengthscale_ratios(scene_scale=7.0)
    for ex in VALID_EXEMPLARS:
        out = ex["output"]
        assert set(out) == RATIO_KEYS, f"unexpected keys: {set(out) ^ RATIO_KEYS}"
        # Ordering constraint and admissible ranges (sanitizer is a no-op on valid input).
        assert out["l_max_ratio"] > out["l_min_ratio"]
        sanitized = sanitize_lengthscale_ratios(out, fallback=fallback)
        for key in ("tau_ratio", "beta", "l_min_ratio", "l_max_ratio", "radius_ratio"):
            assert abs(sanitized[key] - out[key]) < 1e-9, f"{key} was clamped -> out of range in exemplar"


def test_counter_examples_are_actually_bad():
    # Each counter-example illustrates a documented failure and carries an explanation.
    for ex in COUNTER_EXEMPLARS:
        assert ex["why_wrong"]
        assert "bad_output" in ex


def test_few_shot_block_mentions_valid_and_counter():
    block = render_few_shot_block()
    assert "Valid exemplars (20)" in block
    assert "Counter-examples (5)" in block
    assert "Knowledge notes" in block
