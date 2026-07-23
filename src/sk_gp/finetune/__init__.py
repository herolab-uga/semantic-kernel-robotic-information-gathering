"""Scene-LLM QLoRA fine-tuning: policy, contrastive reward, and training loop."""

from .policy import QLoRALengthscalePolicy
from .records import ReplayExample, StepRecord
from .reward import (
    best_candidate_index,
    candidate_softmax,
    contrastive_cross_entropy,
    negative_rmse_reward,
    reward_from_rmse,
)
from .train import perturb_lengthscale_ratios, run_training

__all__ = [
    "QLoRALengthscalePolicy",
    "ReplayExample",
    "StepRecord",
    "negative_rmse_reward",
    "reward_from_rmse",
    "candidate_softmax",
    "contrastive_cross_entropy",
    "best_candidate_index",
    "perturb_lengthscale_ratios",
    "run_training",
]
