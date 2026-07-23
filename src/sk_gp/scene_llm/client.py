"""Scene-LLM client: request kernel parameters from an OpenAI-compatible API.

Thin wrapper that renders the uniform prompt template, calls a chat model with a
JSON-object response format, and returns the parsed configuration.  The OpenAI
SDK is imported lazily; if it is unavailable the client falls back to a raw HTTP
request.  Both ``request_semantic_cfg`` (full config, few-shot) and
``request_lengthscale_ratios`` (fine-tuned ratio task) degrade gracefully to
sanitized defaults on any failure, so the pipeline never crashes on a bad LLM
response -- the formal verifier is the authority on admissibility.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .config import (
    default_lengthscale_ratios,
    sanitize_lengthscale_ratios,
    sanitize_semantic_cfg,
    semantic_cfg_to_lengthscale_ratios,
)
from .prompts import (
    LENGTHSCALE_SYSTEM_PROMPT,
    SEMANTIC_CFG_SYSTEM_PROMPT,
    build_scene_prompt,
    build_semantic_cfg_prompt,
)

__all__ = ["load_api_key", "call_llm_json", "request_semantic_cfg", "request_lengthscale_ratios"]


def load_api_key(api_key_path: str | Path) -> str:
    path = Path(api_key_path)
    if not path.exists():
        raise FileNotFoundError(f"API key file not found: {path}")
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"API key file is empty: {path}")
    os.environ["OPENAI_API_KEY"] = key
    return key


def call_llm_json(messages: List[Dict[str, str]], model: str, api_key: str, temperature: float = 0.1) -> Dict[str, Any]:
    """Call a chat model and parse a single JSON object from its reply."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        body = json.dumps(
            {"model": model, "messages": messages, "response_format": {"type": "json_object"}, "temperature": temperature}
        ).encode("utf-8")
        request = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=body, method="POST")
        request.add_header("Content-Type", "application/json")
        request.add_header("Authorization", f"Bearer {api_key}")
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return json.loads(payload["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM HTTP error {exc.code}: {detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc


def request_semantic_cfg(
    scene_json: Dict[str, Any],
    api_key_path: str | Path,
    *,
    model: str = "gpt-4o-mini",
    scene_scale: float,
    gate_distance_scale: float,
    materials: Sequence[str],
) -> Dict[str, Any]:
    """Few-shot request for a full semantic-kernel config (sanitized).

    Uses the curated few-shot exemplars in :mod:`sk_gp.scene_llm.exemplars` (the FS
    variant of the paper). Requires an API key in ``api_key_path``.
    """
    api_key = load_api_key(api_key_path)
    user_prompt = build_semantic_cfg_prompt(
        scene_json, scene_scale=scene_scale, gate_distance_scale=gate_distance_scale, materials=list(materials)
    )
    raw = call_llm_json(
        [{"role": "system", "content": SEMANTIC_CFG_SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
        model=model,
        api_key=api_key,
    )
    return sanitize_semantic_cfg(raw)


def request_lengthscale_ratios(
    scene_json: Dict[str, Any],
    api_key_path: str | Path,
    *,
    model: str = "gpt-4o",
    complexity: str = "medium",
    scene_scale: float,
    fallback_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Request scene-scale-relative lengthscale ratios (few-shot / inference task).

    Uses the curated few-shot exemplars (:mod:`sk_gp.scene_llm.exemplars`). This is
    the FS variant; the fine-tuned (FT) variant instead samples from a trained
    :class:`~sk_gp.finetune.QLoRALengthscalePolicy`.
    """
    fallback = fallback_cfg or default_lengthscale_ratios(scene_scale)
    try:
        api_key = load_api_key(api_key_path)
        user_prompt = build_scene_prompt(scene_json, complexity=complexity, scene_scale=scene_scale)
        raw = call_llm_json(
            [{"role": "system", "content": LENGTHSCALE_SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
            model=model,
            api_key=api_key,
        )
        return sanitize_lengthscale_ratios(raw, fallback=fallback)
    except Exception:
        return sanitize_lengthscale_ratios(
            {**fallback, "reasoning_summary": "Fallback after LLM request failure."}, fallback=fallback
        )
