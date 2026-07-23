"""Learned-kernel baselines: RBF, Attentive Kernel (AK) and Deep Kernel Learning (DKL).

These require the optional ``baselines`` extra (torch + gpytorch).  Symbols are
imported lazily so that ``import sk_gp`` works without those heavy dependencies;
accessing a baseline class triggers the import (and a helpful error if missing).

    from sk_gp.baselines import AKGPRModel, DKLGPRModel, RBFGPRModel
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "RBFGPRModel",
    "AttentiveKernel",
    "AKGPRModel",
    "DKLGPRModel",
    "FeatureExtractor",
    "train_exact_gp",
    "predict_posterior",
    "ak_effective_lengthscale_map",
]

_LOCATIONS = {
    "RBFGPRModel": "attentive_kernel",
    "AttentiveKernel": "attentive_kernel",
    "AKGPRModel": "attentive_kernel",
    "train_exact_gp": "attentive_kernel",
    "predict_posterior": "attentive_kernel",
    "ak_effective_lengthscale_map": "attentive_kernel",
    "DKLGPRModel": "deep_kernel",
    "FeatureExtractor": "deep_kernel",
}


def __getattr__(name: str) -> Any:
    if name in _LOCATIONS:
        module = importlib.import_module(f".{_LOCATIONS[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
