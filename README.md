<div align="center">

# SK: Semantic Kernel for Robotic Information Gathering

**A Semantic Kernel (SK) for Gaussian Processes that integrates scene semantics into Environmental Field Estimation.**

Sai Krishna Ghanta · Ramviyas Parasuraman
Heterogeneous Robotics Lab (HeRoLab), School of Computing, University of Georgia

---

## Abstract

Environmental spatial phenomena exhibit rich semantic and structural relationships,
yet most robotic information gathering methods assume stationarity and ignore scene
context. We propose a **Semantic Kernel (SK)** for Gaussian Processes (GPs) that
leverages a **Large Language Model (Scene-LLM)** to interpret metric-semantic scene
graphs and extract structural relationships, which are used to construct spatially
varying priors — yielding a **non-stationary GP** that adapts to scene context. A
hybrid **few-shot + QLoRA** approach adapts kernel parameterization from semantics
with minimal supervision. SK improves reconstruction accuracy, sample efficiency and
robustness to noise over state-of-the-art learned kernels (AK, DKL) while avoiding
their data-intensive training — evaluated in simulation and real-world Wi-Fi RSSI
information gathering.

> This repository is a clean, faithful re-implementation of the paper. See
> [`docs/method.md`](docs/method.md) for the full equation-by-equation code map and
> [`Paper.pdf`](Paper.pdf) for the manuscript.


## Key ideas

- **Semantics → kernel, not data → kernel.** SK reads heterogeneity from the scene
  graph rather than inferring it from noisy measurements, making it robust in sparse
  and noisy regimes where AK/DKL struggle.
- **Four non-stationarities** parameterized by the Scene-LLM: lengthscale field
  `ℓ(x)`, semantic gate `G(x,x′)`, signal-variance field `σ²(x)`, and noise field
  `σ²_n(x)`.
- **Guardrails for LLM outputs:** a **formal verifier** (Theorem 1) enforces
  admissibility + geometric feasibility, and a **PCA temporal outlier detector**
  (Eq. 6) rejects hallucinated jumps — guaranteeing well-defined GP posteriors.
- **Plug-and-play:** a few-shot variant needs no training; a QLoRA-fine-tuned small
  model gives maximum performance.

## Architecture

```
scene graph (.json) ─► Scene-LLM ─► Γ ─► Formal Verification (Thm 1) ─► Γ_v
                       (few-shot/QLoRA)      │ invalid → re-prompt / fallback
                                             ▼
                              Temporal Outlier Detection (PCA, Eq. 6)
                                             ▼
                 ℓ(x), σ²(x), σ²_n(x), gate G ─► Semantic Kernel K′ (Eq. 7)
                                             ▼
                        GP regression (Eq. 8-10) ─► mean μ, uncertainty σ²
                                             ▼
                        Adaptive Informative Planner (FAP) ─► next measurement
```

Rendered diagrams (Fig. 2 and the Fig. 3 fine-tuning loop) are in
[`docs/figures/architecture.md`](docs/figures/architecture.md).

## Installation

```bash
git clone <your-fork-url> && cd Semantic_Kernel
python -m venv .venv && source .venv/bin/activate

# Core (semantic kernel, GP, simulator, verification, temporal detection)
pip install -e .

# Optional extras
pip install -e ".[baselines]"   # AK / DKL (torch + gpytorch)
pip install -e ".[llm]"         # Scene-LLM few-shot prompting (openai)
pip install -e ".[finetune]"    # QLoRA fine-tuning (transformers, peft, bitsandbytes)
pip install -e ".[dev]"         # pytest
```

The core runs on `numpy / scipy / scikit-learn / shapely / matplotlib` only — no GPU
required. Running the SK kernel additionally needs a Scene-LLM source (the `llm`
extra + an API key for few-shot, or the `finetune` extra to train an adapter).

## Quickstart

> To run the **`sk`** kernel you must supply a Scene-LLM source — few-shot
> (`--scene-llm few-shot` + an `api_key.txt`) or a fine-tuned adapter
> (`--finetuned-adapter <path>`). Without one, `sk` is skipped with a clear message
> (bare defaults are disabled). All baselines/simulator/verifier/planner run with no LLM.

**End-to-end pipeline demo** (renders ground truth + verifier/temporal steps; the SK
step runs if you pass a Scene-LLM source, otherwise it prints the gate message):

```bash
python experiments/demo_pipeline.py --scene data/scenes/house.json --save-plot          # SK step gated
python experiments/demo_pipeline.py --scene data/scenes/house.json --scene-llm few-shot --save-plot   # needs api_key.txt
```

**Reconstruction under a common adaptive planner** (Fig. 5 a–d):

```bash
# baselines (no LLM needed)
python experiments/run_reconstruction.py --scene data/scenes/house.json --kernels rbf --n-samples 40 --save-plots
# SK via few-shot (needs api_key.txt) or your own adapter
python experiments/run_reconstruction.py --scene data/scenes/house.json --kernels sk rbf --scene-llm few-shot
python experiments/run_reconstruction.py --scene data/scenes/house.json --kernels sk --finetuned-adapter runs/finetune/saved_model
# add ak dkl with the `baselines` extra
```

**Noise robustness sweep** (N1 = 5 dB, N2 = 10 dB; Fig. 5 e–f):

```bash
python experiments/run_noise_sweep.py --scene data/scenes/house.json --kernels rbf
```

## Scene-LLM & fine-tuning

The Scene-LLM maps a scene graph to kernel parameters (`docs/method.md`, Section III).
Both variants are runnable from this repo; only the pre-trained adapter weights are
withheld.

- **Few-shot (FS):** a pretrained LLM + the **provided** curated exemplars
  (`sk_gp.scene_llm.exemplars`, 20 valid + 5 counter-examples). Put your key in
  `api_key.txt`, then call `sk_gp.scene_llm.request_lengthscale_ratios` /
  `request_semantic_cfg`, or pass `--scene-llm few-shot` to the experiment scripts.
- **QLoRA fine-tuning (FT)** (Fig. 3) with the paper's setup (r=16, α=32, dropout 0.05,
  AdamW 2e-4, batch 2×8, 5 epochs, seq 4096, K=8 candidates, 4-bit NF4):

```bash
# Reward search only (no GPU / HF stack):
python experiments/run_finetune.py --skip-qlora --n-simple 2 --n-medium 1 --n-complex 1
# Full fine-tuning (needs the `finetune` extra + GPU):
python experiments/run_finetune.py --base-model Qwen/Qwen3-8B --n-simple 60 --n-medium 60 --n-complex 80
```

The loop draws `K` candidate parameterizations per world, scores them by SK
reconstruction RMSE, treats the best as the positive `k*`, and trains the LoRA
adapter with the **contrastive cross-entropy** objective (Eq. 11).

## Repository structure

```
Semantic_Kernel/
├── src/sk_gp/
│   ├── kernel/           # fields (Eq.1-2,5), gate (Eq.3-4), semantic_kernel (Eq.7), gp (Eq.8-10)
│   ├── envs/             # scene-graph Environment + RSSI propagation model (Eq.12)
│   ├── simulator/        # procedural scene generator + radio-world oracle
│   ├── scene_llm/        # prompts, exemplars (20 valid + 5 counter), config schema, client
│   ├── finetune/         # QLoRA policy, contrastive reward (Eq.11), training loop
│   ├── baselines/        # RBF, Attentive Kernel (AK), Deep Kernel Learning (DKL)
│   ├── planning/         # acquisition + adaptive informative sampling (FAP / A-IPP)
│   ├── verification.py   # formal verification of LLM parameters (Theorem 1)
│   └── temporal.py       # PCA temporal outlier detection (Eq.6)
├── data/scenes/          # House (H), School (S, 3 APs), and observed variants
├── experiments/          # demo_pipeline, run_reconstruction, run_noise_sweep, run_finetune
├── configs/              # house.yaml, school.yaml, finetune.yaml
├── docs/                 # method.md (paper↔code map), figures/architecture.md
├── tests/                # kernel, fields, verification, temporal, simulator
└── Paper.pdf
```

## Reproducing paper components

| Paper component | Where |
|---|---|
| Non-stationary Semantic Kernel `K′` | `sk_gp.kernel.SemanticKernel` |
| Formal verification (Theorem 1) | `sk_gp.verification` |
| Temporal outlier detection (Eq. 6) | `sk_gp.temporal` |
| Scene-LLM prompting + QLoRA (Eq. 11) | `sk_gp.scene_llm`, `sk_gp.finetune` |
| RSSI simulator (Eq. 12) | `sk_gp.simulator`, `sk_gp.envs.propagation` |
| Baselines AK / DKL / RBF | `sk_gp.baselines` |
| Adaptive informative planner (FAP) | `sk_gp.planning` |

> **Note.** SK requires a Scene-LLM source (few-shot with the provided exemplars, or
> a fine-tuned adapter you train). The pre-trained adapter is not shipped, and the
> generic offline defaults are deliberately disabled since they are not
> representative of the paper.

## Tests

```bash
pytest -q
```

## Citation

```bibtex
@inproceedings{ghanta_sk_semantic_kernel,
  title     = {SK: Semantic Kernel for Robotic Information Gathering},
  author    = {Ghanta, Sai Krishna and Parasuraman, Ramviyas},
  booktitle = {Accepted for IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2026},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
