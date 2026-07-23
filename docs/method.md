# Method — paper ↔ code map

This document maps every component of **"SK: Semantic Kernel for Robotic
Information Gathering"** to its implementation in `src/sk_gp/`.

## Overview (Fig. 2)

```
scene graph (.json)
      │
      ▼
  Scene-LLM  ──►  parameter set Γ  ──►  Formal Verification (Thm 1)  ──►  Γ_v
 (few-shot / QLoRA)                       │ invalid → re-prompt / fallback
                                          ▼
                          Temporal Outlier Detection (PCA, Eq. 6)
                                          │ outlier → last valid Γ_v
                                          ▼
                     continuous non-stationarity fields ℓ(x), σ²(x), σ²_n(x), gate G
                                          ▼
                        Semantic Kernel K′  ──►  GP regression  ──►  mean μ, uncertainty σ²
```

## Equation ↔ code

| Paper | Concept | Code |
|-------|---------|------|
| Eq. (1)–(2) | Lengthscale field ℓ(x): convex combination of primitives, distance-decayed weights | [`kernel/fields.py`](../src/sk_gp/kernel/fields.py) — `primitive_weights`, `LengthscaleField`, `SemanticLengthscale` |
| Eq. (3) | Geometry-aware attenuation A(x,x′) via signed distance fields + logistic | [`kernel/gate.py`](../src/sk_gp/kernel/gate.py), `SemanticKernel._geometry_attenuation` |
| Eq. (4) | Semantic gate G = (z(x)ᵀz(x′))·A(x,x′) | [`kernel/semantic_kernel.py`](../src/sk_gp/kernel/semantic_kernel.py) `SemanticKernel` + [`hyperfield.py`](../src/sk_gp/kernel/hyperfield.py) `SemanticHyperField.membership` |
| Eq. (5) | Signal/noise variance fields σ²(x), σ²_n(x) | [`kernel/fields.py`](../src/sk_gp/kernel/fields.py) — `VarianceField`, `NoiseField` |
| Eq. (6) | Temporal outlier detection: PCA Mahalanobis + semantic info gain g_t | [`temporal.py`](../src/sk_gp/temporal.py) `TemporalOutlierDetector` |
| Eq. (7) | Semantic Kernel K′ = G·K(·;ℓ,σ²) + σ²_n·δ | [`kernel/semantic_kernel.py`](../src/sk_gp/kernel/semantic_kernel.py) `SemanticKernel.__call__` |
| Eq. (8)–(10) | GP posterior mean / latent / predictive variance | [`kernel/gp.py`](../src/sk_gp/kernel/gp.py) `fit_predict_semantic_kernel` (sklearn GPR) |
| Theorem 1 | Formal verification over Θ_adm ∩ C, failure log, fallback | [`verification.py`](../src/sk_gp/verification.py) `verify_parameters`, `FormalVerifier` |
| Eq. (11) | Contrastive cross-entropy over K candidates (softmax, temperature τ) | [`finetune/reward.py`](../src/sk_gp/finetune/reward.py) `candidate_softmax`, `contrastive_cross_entropy` |
| Eq. (12) | Semantics-aware RSSI propagation (log-distance + material attenuation + fading) | [`envs/propagation.py`](../src/sk_gp/envs/propagation.py), [`simulator/radio_world.py`](../src/sk_gp/simulator/radio_world.py) |

## Scene-LLM (Section III, Fig. 3)

* **Prompts** — uniform template (role, task contract, JSON output schema, few-shot
  exemplars with knowledge notes): [`scene_llm/prompts.py`](../src/sk_gp/scene_llm/prompts.py).
* **Few-shot exemplars** — the curated 20 valid + 5 counter-examples (Section III):
  [`scene_llm/exemplars.py`](../src/sk_gp/scene_llm/exemplars.py), embedded into both
  prompt builders.
* **Config schema + admissibility** — [`scene_llm/config.py`](../src/sk_gp/scene_llm/config.py)
  (`sanitize_semantic_cfg`, ratio↔cfg conversions).
* **Client** — [`scene_llm/client.py`](../src/sk_gp/scene_llm/client.py) (few-shot `request_semantic_cfg`,
  fine-tuned `request_lengthscale_ratios`).
* **QLoRA policy** — [`finetune/policy.py`](../src/sk_gp/finetune/policy.py): frozen 4-bit
  backbone, LoRA adapters (r=16, α=32, dropout 0.05) on attention + FFN projections.
* **Training loop** — [`finetune/train.py`](../src/sk_gp/finetune/train.py): K candidates per
  world → SK reconstruction reward (−RMSE) → best candidate is the positive `k*` →
  contrastive CE (Eq. 11) → distill positives into the adapter.

## Experiments (Section IV)

* **Environments** — House (H) and School (S, 3 fused APs) in [`data/scenes/`](../data/scenes/);
  procedural training worlds via [`simulator/generator.py`](../src/sk_gp/simulator/generator.py).
* **Baselines** — RBF (sklearn), AK & DKL (GPyTorch) in [`baselines/`](../src/sk_gp/baselines/).
* **Planner (FAP / A-IPP)** — [`planning/informative_sampling.py`](../src/sk_gp/planning/informative_sampling.py),
  acquisition in [`planning/acquisition.py`](../src/sk_gp/planning/acquisition.py).
* **Metrics** — RMSE of the posterior mean vs. ground truth and mean posterior
  uncertainty (Fig. 5 a–d); noise sweep N1/N2 (Fig. 5 e–f); Executability via the
  formal verifier (Fig. 5 g–h).

## Notes on faithfulness

The kernel used throughout the running pipeline is the non-stationary **Gibbs**
generalization of the RBF with a spatially varying lengthscale and the semantic
gate (Eq. 7). The signal/noise **variance fields** (Eq. 5) are implemented as
first-class `PrimitiveField`s and can be attached to `SemanticKernel` via
`noise_field`; the reconstruction experiments default to a scalar signal variance
plus a `WhiteKernel`/`alpha` noise term for parity with the original code. To
reproduce the paper's reported gains you must supply Scene-LLM parameters — either
the few-shot variant (using the provided exemplars in
[`scene_llm/exemplars.py`](../src/sk_gp/scene_llm/exemplars.py) + an API key) or a
QLoRA adapter you train with [`sk_gp.finetune`](../src/sk_gp/finetune/). The
pre-trained adapter is not distributed, and the generic offline defaults are
deliberately disabled for SK inference (see [`_release.py`](../src/sk_gp/_release.py)).
