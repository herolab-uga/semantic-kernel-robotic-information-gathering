"""Rewards and the contrastive cross-entropy objective (Eq. 11).

For each environment ``i`` the Scene-LLM draws ``K`` candidate parameterizations
``Gamma_i^k ~ p_theta(Gamma | G_i)``.  Each candidate instantiates the Semantic
Kernel, GP inference reconstructs the field, and the reward is the negative RMSE
(normalized by environment size).  The highest-reward candidate ``k*`` is the
positive example.

The adapter assigns a score ``s_theta(Gamma_i^k, G_i)`` to each candidate,
inducing a categorical distribution via a temperature-``tau`` softmax (Eq. 11)::

    pi_theta(k | G_i) = softmax_k( s_theta(Gamma_i^k, G_i) / tau )

and is trained with the contrastive cross-entropy loss that concentrates
probability on the best candidate::

    L(theta) = - sum_i log pi_theta(k*_i | G_i)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = [
    "reward_from_rmse",
    "negative_rmse_reward",
    "candidate_softmax",
    "contrastive_cross_entropy",
]


def negative_rmse_reward(rmse: float, env_size: float = 1.0) -> float:
    """Reward as negative RMSE per unit environment size (paper's base reward)."""
    return -float(rmse) / max(float(env_size), 1e-6)


def reward_from_rmse(*, candidate_rmse: float, default_sk_rmse: float, ak_rmse: float, ak_beat_bonus: float = 0.75) -> float:
    """Shaped reward: negative RMSE plus improvement over the default SK and a
    bonus for beating the AK baseline (used by the reward-guided search loop)."""
    reward = -float(candidate_rmse)
    reward += max(0.0, default_sk_rmse - candidate_rmse)
    if candidate_rmse < ak_rmse:
        reward += float(ak_beat_bonus) + (ak_rmse - candidate_rmse)
    return float(reward)


def candidate_softmax(scores: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    """Categorical distribution ``pi_theta(k | G_i)`` over candidates (Eq. 11)."""
    scores = np.asarray(scores, dtype=np.float64) / max(float(temperature), 1e-6)
    scores -= scores.max()
    exp = np.exp(scores)
    return exp / (exp.sum() + 1e-12)


def contrastive_cross_entropy(scores: Sequence[float], best_index: int, temperature: float = 1.0) -> float:
    """Contrastive cross-entropy loss ``-log pi_theta(k* | G_i)`` (Eq. 11)."""
    pi = candidate_softmax(scores, temperature=temperature)
    return float(-np.log(pi[int(best_index)] + 1e-12))


def best_candidate_index(rewards: Sequence[float]) -> int:
    """Index ``k*`` of the highest-reward candidate."""
    return int(np.argmax(np.asarray(rewards, dtype=np.float64)))
