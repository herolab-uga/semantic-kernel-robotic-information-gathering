"""Release gating for the Semantic Kernel.

This public release ships **everything needed to run and reproduce SK** except the
pre-trained fine-tuned adapter weights:

* the full SK architecture (kernel, verifier, temporal detector, simulator, planner),
* the complete QLoRA fine-tuning pipeline (:mod:`sk_gp.finetune`),
* the curated few-shot exemplars (:mod:`sk_gp.scene_llm.exemplars`).

So there are two supported ways to drive SK, matching the paper's two variants:

* **Few-shot (FS):** a pretrained LLM + the provided exemplars (needs an API key).
* **Fine-tuned (FT):** train your own QLoRA adapter here, then load it.

What is **not** representative is running SK with the *generic offline defaults*
and no Scene-LLM at all — that does not reproduce the paper and can underperform a
stationary RBF. To avoid presenting misleading numbers, the SK inference entry
points require a Scene-LLM parameter *source* (a few-shot/fine-tuned config or an
adapter path). Nothing is hidden or silently broken — the requirement is explicit.
"""

from __future__ import annotations

# Set to True only in a build that bundles a pre-trained fine-tuned adapter.
FINETUNED_ADAPTER_INCLUDED = False

_MESSAGE = (
    "The Semantic Kernel must be driven by Scene-LLM parameters (paper Section III).\n"
    "Provide one of:\n"
    "  - a few-shot config from `sk_gp.scene_llm.request_lengthscale_ratios` / "
    "`request_semantic_cfg` (uses the provided exemplars + an API key), or\n"
    "  - a fine-tuned adapter you train with `sk_gp.finetune` "
    "(pass `finetuned_adapter=...` / `--finetuned-adapter`).\n"
    "Running SK with the generic offline defaults is not representative of the paper "
    "and is therefore disabled by default."
)


class SceneLLMParametersRequired(RuntimeError):
    """Raised when an SK inference path is used without a Scene-LLM parameter source."""


# Backwards-compatible alias.
FineTunedModelRequired = SceneLLMParametersRequired


def require_scene_llm_parameters(source=None, *, allow_untuned: bool = False) -> None:
    """Guard SK inference / reproduction entry points.

    Passes when a Scene-LLM parameter ``source`` is supplied (a few-shot/fine-tuned
    config dict, an adapter path, or any truthy marker), when the build bundles a
    fine-tuned adapter, or when ``allow_untuned=True`` is explicitly set (used
    internally by the fine-tuning reward loop, which is *producing* the adapter).
    Otherwise raises :class:`SceneLLMParametersRequired`.
    """
    if allow_untuned or FINETUNED_ADAPTER_INCLUDED or source is not None:
        return
    raise SceneLLMParametersRequired(_MESSAGE)
