"""Reconstruction experiment: SK vs baseline kernels under a common planner.

Loads a scene, synthesizes the semantics-aware RSSI ground truth, and runs the
adaptive informative-sampling loop (FAP / A-IPP) once per kernel, logging RMSE and
mean posterior uncertainty against the ground truth at every sample.  This
reproduces the convergence comparison of Fig. 5 (a)-(d).

Always-available kernels: ``sk`` (Semantic Kernel) and ``rbf`` (stationary GP).
``ak`` (Attentive Kernel) and ``dkl`` (Deep Kernel Learning) require the
``baselines`` extra (torch + gpytorch) and are skipped with a warning if missing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

from ._release import SceneLLMParametersRequired, require_scene_llm_parameters
from .envs import best_ap_field, clip_dbm, compute_grid, load_environment
from .kernel import (
    SemanticGibbsRBF,
    SemanticHyperField,
    SemanticLengthscale,
    build_gate_entities,
    resolve_gate_material_params,
)
from .kernel.semantic_kernel import SemanticKernel
from .planning import run_informative_sampling
from .scene_llm import (
    build_world_scaled_semantic_cfg,
    default_lengthscale_ratios,
    default_semantic_cfg,
    lengthscale_ratios_to_semantic_cfg,
    load_api_key,
    request_lengthscale_ratios,
    sanitize_semantic_cfg,
)

__all__ = ["reconstruction_main", "make_fit_gp"]


def _scene_scale(env) -> tuple[float, float]:
    x_span = env.bounds[1] - env.bounds[0]
    y_span = env.bounds[3] - env.bounds[2]
    return 0.75 * math.sqrt(x_span * y_span), 0.5 * min(x_span, y_span)


def _resolve_sk_config(env, *, scene_scale: float, scene_llm_mode: str, finetuned_adapter, scene_json, api_key_path):
    """Obtain the SK config (Scene-LLM source) that drives the kernel.

    Returns ``(sem_cfg, source)``. ``source`` is a truthy marker satisfying the
    Scene-LLM requirement (few-shot config or adapter path); ``None`` means "run
    with generic defaults", which is rejected by the gate.
    """
    if finetuned_adapter is not None:
        # Fine-tuned (FT): sample lengthscale ratios from the trained adapter.
        from .finetune import QLoRALengthscalePolicy

        policy = QLoRALengthscalePolicy(adapter_source=finetuned_adapter, trainable=False)
        ratios = policy.propose(
            prompt_text=json.dumps(scene_json, separators=(",", ":")),
            fallback_cfg=default_lengthscale_ratios(scene_scale),
        )
        return lengthscale_ratios_to_semantic_cfg(ratios, scene_scale=scene_scale), finetuned_adapter

    if scene_llm_mode == "few-shot":
        # Few-shot (FS): pretrained LLM + the provided exemplars (needs an API key).
        load_api_key(api_key_path)  # clear error if the key file is missing/empty
        ratios = request_lengthscale_ratios(scene_json, api_key_path, scene_scale=scene_scale)
        return lengthscale_ratios_to_semantic_cfg(ratios, scene_scale=scene_scale), ratios

    # No Scene-LLM source -> gate raises with guidance.
    require_scene_llm_parameters(None)


def make_fit_gp(
    kernel_name: str,
    env,
    *,
    noise_std_db: float,
    domain_diag: float,
    signal_var: float,
    finetuned_adapter=None,
    scene_llm_mode: str = "none",
    scene_json=None,
    api_key_path: str = "api_key.txt",
) -> Callable:
    """Build a ``fit_gp(X_train, y_train) -> fitted GP`` for the requested kernel.

    The ``sk`` path must be driven by a Scene-LLM parameter source: either the
    few-shot variant (``scene_llm_mode="few-shot"`` + the provided exemplars + an
    API key) or a fine-tuned adapter you trained (``finetuned_adapter``). Without a
    source it raises ``SceneLLMParametersRequired`` (see :mod:`sk_gp._release`).
    """
    noise_var = max(noise_std_db ** 2, 0.35 ** 2)
    scene_scale, gate_distance_scale = _scene_scale(env)

    if kernel_name == "rbf":
        length_scale = max(0.14 * domain_diag, 1e-3)

        def fit_gp(X_train, y_train):
            kernel = (
                C(signal_var, constant_value_bounds="fixed") * RBF(length_scale=length_scale, length_scale_bounds="fixed")
                + WhiteKernel(noise_level=noise_var, noise_level_bounds="fixed")
            )
            gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, optimizer=None, normalize_y=True)
            gp.fit(X_train, y_train)
            return gp

        return fit_gp

    if kernel_name == "sk":
        # SK must be driven by a Scene-LLM parameter source (few-shot or fine-tuned);
        # this resolves it and raises SceneLLMParametersRequired if neither is given.
        cfg, _source = _resolve_sk_config(
            env, scene_scale=scene_scale, scene_llm_mode=scene_llm_mode,
            finetuned_adapter=finetuned_adapter, scene_json=scene_json, api_key_path=api_key_path,
        )
        sem_l = SemanticLengthscale(
            env, tau=cfg["tau"], beta=cfg["beta"], l_min=cfg["l_min"], l_max=cfg["l_max"], radius=cfg["radius"]
        )
        gate_material_params = cfg.get("gate_material_params") or {}
        gate_entities = build_gate_entities(env, resolve_gate_material_params(env, gate_material_params))

        def fit_gp(X_train, y_train):
            hf = SemanticHyperField(
                sem_l, K=cfg["K"], temperature=cfg["temperature"], center_mode=cfg["center_mode"], priors=cfg["priors"]
            )
            hf.fit_centers(X_train, mode=hf.center_mode)
            kernel = SemanticKernel(
                hyperfield=hf, gate_entities=gate_entities, gate_distance_scale=gate_distance_scale,
                variance=cfg["sem_variance"], jitter=cfg["sem_jitter"],
            ) + WhiteKernel(noise_level=noise_var, noise_level_bounds="fixed")
            gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, optimizer=None, normalize_y=True)
            gp.fit(X_train, y_train)
            return gp

        return fit_gp

    if kernel_name in ("ak", "dkl"):
        return _make_gpytorch_fit_gp(kernel_name, env)

    raise ValueError(f"Unknown kernel '{kernel_name}'. Choose from sk, rbf, ak, dkl.")


def _make_gpytorch_fit_gp(kernel_name: str, env) -> Callable:
    """Adapter exposing a sklearn-like ``predict(X, return_std=True)`` for AK/DKL."""
    import torch

    from .baselines import AKGPRModel, DKLGPRModel, predict_posterior, train_exact_gp

    xmin, xmax, ymin, ymax = env.bounds
    x_span, y_span = max(xmax - xmin, 1e-9), max(ymax - ymin, 1e-9)

    def _norm(X):
        return np.column_stack([(X[:, 0] - xmin) / x_span, (X[:, 1] - ymin) / y_span]).astype(np.float32)

    class _Wrapper:
        def __init__(self, model, likelihood):
            self.model = model
            self.likelihood = likelihood

        def predict(self, X, return_std=False):
            import gpytorch  # noqa: F401

            tx = torch.tensor(_norm(np.atleast_2d(X)), dtype=torch.float32)
            mean, var = predict_posterior(self.model, tx)
            if return_std:
                return mean, np.sqrt(np.maximum(var, 0.0))
            return mean

    def fit_gp(X_train, y_train):
        tx = torch.tensor(_norm(X_train), dtype=torch.float32)
        ty = torch.tensor(np.asarray(y_train, dtype=np.float32))
        noise = torch.full((tx.shape[0],), 1e-4, dtype=torch.float32)
        likelihood = _gpytorch_likelihood(noise)
        if kernel_name == "ak":
            model = AKGPRModel(tx, ty, likelihood, input_dim=2, m_kernels=10)
        else:
            model = DKLGPRModel(tx, ty, likelihood, input_dim=2)
        train_exact_gp(model, likelihood, tx, ty, steps=90, lr=0.03)
        return _Wrapper(model, likelihood)

    return fit_gp


def _gpytorch_likelihood(noise):
    import gpytorch

    return gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noise)


def reconstruction_main(argv: List[str] | None = None) -> Dict:
    parser = argparse.ArgumentParser(description="SK vs baseline reconstruction under adaptive informative sampling.")
    parser.add_argument("--scene", default="data/scenes/house.json")
    parser.add_argument("--kernels", nargs="+", default=["sk", "rbf"], help="Subset of: sk rbf ak dkl")
    parser.add_argument("--n-samples", type=int, default=40)
    parser.add_argument("--noise-db", type=float, default=5.0, help="Measurement noise (5=N1, 10=N2).")
    parser.add_argument("--res", type=int, default=60)
    parser.add_argument("--p-tx", type=float, default=-30.0)
    parser.add_argument("--n-exp", type=float, default=3.0)
    parser.add_argument("--candidate-stride", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="runs/reconstruction")
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument(
        "--scene-llm",
        choices=["none", "few-shot"],
        default="none",
        help="Scene-LLM source for the `sk` kernel: 'few-shot' uses the provided "
        "exemplars via the API (needs --api-key-path). Use --finetuned-adapter for the "
        "fine-tuned (FT) variant.",
    )
    parser.add_argument("--api-key-path", default="api_key.txt", help="API key file for the few-shot Scene-LLM.")
    parser.add_argument(
        "--finetuned-adapter",
        default=None,
        help="Path to a Scene-LLM adapter you fine-tuned with sk_gp.finetune (FT variant). "
        "The pre-trained adapter is not shipped in this release.",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.scene, "r") as f:
        scene_json = json.load(f)
    env = load_environment(args.scene)
    X, Y, positions = compute_grid(env, nx=args.res, ny=args.res)
    y_true = clip_dbm(best_ap_field(env, positions, P_tx=args.p_tx, n_exp=args.n_exp))
    domain_diag = float(np.hypot(env.bounds[1] - env.bounds[0], env.bounds[3] - env.bounds[2]))
    signal_var = max(float(np.var(y_true)), 1e-3)

    summary: Dict[str, Dict] = {}
    curves: Dict[str, List[float]] = {}
    for kernel_name in args.kernels:
        try:
            fit_gp = make_fit_gp(
                kernel_name, env, noise_std_db=args.noise_db, domain_diag=domain_diag,
                signal_var=signal_var, finetuned_adapter=args.finetuned_adapter,
                scene_llm_mode=args.scene_llm, scene_json=scene_json, api_key_path=args.api_key_path,
            )
        except (ImportError, SceneLLMParametersRequired, FileNotFoundError) as exc:
            print(f"[skip] {kernel_name}: {exc}")
            continue
        result = run_informative_sampling(
            positions, y_true, (args.res, args.res), fit_gp,
            n_samples=args.n_samples, candidate_stride=args.candidate_stride,
            noise_std_db=args.noise_db, seed=args.seed,
        )
        final_rmse = result.step_rmse[-1] if result.step_rmse else float("nan")
        summary[kernel_name] = {"final_rmse": final_rmse, "final_avg_std": result.step_avg_std[-1] if result.step_avg_std else float("nan")}
        curves[kernel_name] = result.step_rmse
        print(f"[{kernel_name:4s}] final RMSE over {args.n_samples} samples (noise {args.noise_db} dB): {final_rmse:.3f}")

    (out_dir / "summary.json").write_text(json.dumps({"scene": args.scene, "noise_db": args.noise_db, "results": summary}, indent=2))

    if args.save_plots and curves:
        _plot_rmse_curves(curves, out_dir / "rmse_vs_samples.png", noise_db=args.noise_db)
        print(f"[plot] {out_dir / 'rmse_vs_samples.png'}")

    return summary


def _plot_rmse_curves(curves: Dict[str, List[float]], path: Path, *, noise_db: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    for name, rmse in curves.items():
        plt.plot(range(1, len(rmse) + 1), rmse, label=name.upper(), linewidth=2)
    plt.xlabel("# samples")
    plt.ylabel("RMSE (dB)")
    plt.title(f"Reconstruction RMSE vs samples (noise {noise_db} dB)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


if __name__ == "__main__":
    reconstruction_main()
