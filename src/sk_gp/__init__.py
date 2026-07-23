"""SK: Semantic Kernel for Robotic Information Gathering.

A Semantic Kernel (SK) for Gaussian Processes that integrates scene semantics
into environmental field estimation.  A (optionally fine-tuned) Scene-LLM
interprets a metric-semantic scene graph and emits parameters for a non-stationary
kernel; formal verification and temporal outlier detection guard the parameters
before GP inference produces calibrated mean/uncertainty fields.

Subpackages
-----------
``kernel``       Non-stationarity fields, semantic gate, kernels and GP inference.
``envs``         Scene-graph environment container and RSSI propagation model.
``simulator``    Procedural scene generator and radio-world oracle.
``scene_llm``    Prompts, config schema, and the LLM client.
``finetune``     QLoRA fine-tuning of the Scene-LLM (contrastive objective).
``baselines``    RBF / Attentive Kernel / Deep Kernel Learning (optional: torch).
``planning``     Adaptive informative sampling (FAP / A-IPP).
``verification`` Formal verification of LLM parameters (Theorem 1).
``temporal``     PCA temporal outlier detection over parameter fields (Eq. 6).
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import envs, kernel, planning, scene_llm, simulator, temporal, verification
from .kernel import (
    SemanticGatedGibbs,
    SemanticGibbsRBF,
    SemanticHyperField,
    SemanticKernel,
    SemanticLengthscale,
    build_gate_entities,
    fit_predict_semantic_kernel,
)
from .temporal import TemporalOutlierDetector
from .verification import FormalVerifier, verify_parameters

__all__ = [
    "__version__",
    "envs",
    "kernel",
    "simulator",
    "scene_llm",
    "planning",
    "verification",
    "temporal",
    "SemanticKernel",
    "SemanticGatedGibbs",
    "SemanticGibbsRBF",
    "SemanticHyperField",
    "SemanticLengthscale",
    "build_gate_entities",
    "fit_predict_semantic_kernel",
    "FormalVerifier",
    "verify_parameters",
    "TemporalOutlierDetector",
]
