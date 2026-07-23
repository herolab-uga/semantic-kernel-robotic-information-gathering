"""Scene-LLM: prompts, configuration schema, and API client."""

from .client import (
    call_llm_json,
    load_api_key,
    request_lengthscale_ratios,
    request_semantic_cfg,
)
from .config import (
    build_world_scaled_semantic_cfg,
    default_lengthscale_ratios,
    default_semantic_cfg,
    lengthscale_ratios_to_semantic_cfg,
    sanitize_lengthscale_ratios,
    sanitize_semantic_cfg,
    semantic_cfg_to_lengthscale_ratios,
)
from .exemplars import (
    COUNTER_EXEMPLARS,
    KNOWLEDGE_NOTES,
    VALID_EXEMPLARS,
    iter_valid_outputs,
    render_few_shot_block,
)
from .prompts import (
    LENGTHSCALE_SYSTEM_PROMPT,
    SEMANTIC_CFG_SYSTEM_PROMPT,
    build_scene_prompt,
    build_semantic_cfg_prompt,
)

__all__ = [
    "load_api_key",
    "call_llm_json",
    "request_semantic_cfg",
    "request_lengthscale_ratios",
    "default_semantic_cfg",
    "sanitize_semantic_cfg",
    "build_world_scaled_semantic_cfg",
    "default_lengthscale_ratios",
    "sanitize_lengthscale_ratios",
    "lengthscale_ratios_to_semantic_cfg",
    "semantic_cfg_to_lengthscale_ratios",
    "LENGTHSCALE_SYSTEM_PROMPT",
    "SEMANTIC_CFG_SYSTEM_PROMPT",
    "build_scene_prompt",
    "build_semantic_cfg_prompt",
    "VALID_EXEMPLARS",
    "COUNTER_EXEMPLARS",
    "KNOWLEDGE_NOTES",
    "render_few_shot_block",
    "iter_valid_outputs",
]
