"""Dataclasses shared across the fine-tuning loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

__all__ = ["StepRecord", "ReplayExample"]


@dataclass
class StepRecord:
    """One reward-search step within a training environment."""

    env_idx: int
    complexity: str
    step: int
    proposal_source: str
    reward: float
    rmse_candidate: float
    rmse_sk_default: float
    rmse_ak: float
    best_rmse_so_far: float
    beat_ak: bool
    proposal: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplayExample:
    """A (prompt, best-answer) pair distilled from the highest-reward candidate."""

    env_idx: int
    complexity: str
    reward: float
    prompt_text: str
    answer_json: Dict[str, Any]
