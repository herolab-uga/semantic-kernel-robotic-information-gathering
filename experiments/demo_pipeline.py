#!/usr/bin/env python3
"""End-to-end Semantic Kernel pipeline demo (Fig. 2).

Walks the full inference path on one scene, with an offline fallback config so it
runs with no API key or GPU:

    scene graph -> (Scene-LLM cfg) -> formal verification (Theorem 1)
                -> temporal outlier check (Eq. 6) -> Semantic Kernel GP
                -> mean + uncertainty fields.

    python experiments/demo_pipeline.py --scene data/scenes/house.json --save-plot
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sk_gp._release import SceneLLMParametersRequired  # noqa: E402
from sk_gp.envs import best_ap_field, clip_dbm, compute_grid, load_environment, sample_training_data  # noqa: E402
from sk_gp.kernel import (  # noqa: E402
    SemanticHyperField,
    SemanticLengthscale,
    build_gate_entities,
    fit_predict_semantic_kernel,
    resolve_gate_material_params,
)
from sk_gp.scene_llm import (  # noqa: E402
    build_world_scaled_semantic_cfg,
    default_semantic_cfg,
    lengthscale_ratios_to_semantic_cfg,
    request_lengthscale_ratios,
    sanitize_semantic_cfg,
)
from sk_gp.temporal import TemporalOutlierDetector  # noqa: E402
from sk_gp.verification import AdmissibleBounds, verify_parameters  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end Semantic Kernel demo.")
    parser.add_argument("--scene", default="data/scenes/house.json")
    parser.add_argument("--res", type=int, default=60)
    parser.add_argument("--n-samples", type=int, default=40)
    parser.add_argument("--noise-db", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-plot", action="store_true")
    parser.add_argument("--out", default="runs/demo/pipeline.png")
    parser.add_argument(
        "--scene-llm",
        choices=["none", "few-shot"],
        default="none",
        help="Scene-LLM source: 'few-shot' uses the provided exemplars via the API "
        "(needs --api-key-path). Otherwise the SK step is gated.",
    )
    parser.add_argument("--api-key-path", default="api_key.txt")
    parser.add_argument(
        "--finetuned-adapter",
        default=None,
        help="Path to a Scene-LLM adapter you fine-tuned with sk_gp.finetune (FT variant).",
    )
    args = parser.parse_args()

    # 1) Scene graph -> environment + ground truth.
    with open(args.scene, "r") as f:
        scene_json = json.load(f)
    env = load_environment(args.scene)
    X, Y, positions = compute_grid(env, nx=args.res, ny=args.res)
    y_true = clip_dbm(best_ap_field(env, positions))
    Xtr, ytr, _ = sample_training_data(positions, y_true, num_samples=args.n_samples, noise_std=args.noise_db, seed=args.seed)
    print(f"[scene] {Path(args.scene).name}: rooms={len(env.rooms)} walls={len(env.walls)} objects={len(env.objects)}")

    # 2) Scene-LLM config -> world-scaled parameters.
    #    few-shot: query the LLM with the provided exemplars; otherwise generic
    #    defaults are used only for the (illustrative) verify/temporal steps, and
    #    the SK inference step below is gated.
    x_span, y_span = env.bounds[1] - env.bounds[0], env.bounds[3] - env.bounds[2]
    scene_scale = 0.75 * math.sqrt(x_span * y_span)
    gate_distance_scale = 0.5 * min(x_span, y_span)
    scene_llm_source = args.finetuned_adapter
    if args.scene_llm == "few-shot":
        ratios = request_lengthscale_ratios(scene_json, args.api_key_path, scene_scale=scene_scale)
        cfg = lengthscale_ratios_to_semantic_cfg(ratios, scene_scale=scene_scale)
        scene_llm_source = ratios
    else:
        cfg = build_world_scaled_semantic_cfg(sanitize_semantic_cfg(default_semantic_cfg()), scene_scale)

    # 3) Formal verification (Theorem 1).
    gate_entities = build_gate_entities(env, resolve_gate_material_params(env, {}))
    result = verify_parameters(cfg, bounds=AdmissibleBounds(), domain=env.bounds, entities=gate_entities)
    print(f"[verify] valid={result.valid} failures={result.failures}")
    if not result.valid:
        raise SystemExit("Parameters failed verification; a real pipeline would re-prompt or fall back.")

    # 4) Temporal outlier check on the induced lengthscale field (Eq. 6).
    sem_l = SemanticLengthscale(env, tau=cfg["tau"], beta=cfg["beta"], l_min=cfg["l_min"], l_max=cfg["l_max"], radius=cfg["radius"])
    detector = TemporalOutlierDetector(window=5, tau0=9.0, n_components=2)
    n_entities = len(env.walls) + len(env.objects)
    decision = detector.update(sem_l(positions[::20]), n_entities=n_entities)
    print(f"[temporal] first-frame outlier={decision.is_outlier} (history warming up)")

    # 5) Semantic Kernel GP inference -> mean + uncertainty.
    #    Gated: requires a fine-tuned Scene-LLM adapter (not shipped in this release).
    hf = SemanticHyperField(sem_l, K=cfg["K"], temperature=cfg["temperature"], center_mode=cfg["center_mode"], priors=cfg["priors"])
    try:
        mean, std, _ = fit_predict_semantic_kernel(
            Xtr, ytr, positions, hyperfield=hf, gate_entities=gate_entities,
            gate_distance_scale=gate_distance_scale, variance=cfg["sem_variance"], jitter=cfg["sem_jitter"],
            scene_llm_source=scene_llm_source,
        )
    except SceneLLMParametersRequired as exc:
        print("\n[SK gated] " + str(exc))
        raise SystemExit(2)
    mean = np.clip(mean, -50.0, 0.0)
    rmse = float(np.sqrt(np.mean((mean - y_true) ** 2)))
    print(f"[sk-gp] RMSE={rmse:.3f} dB | mean uncertainty={float(np.mean(std)):.3f}")

    if args.save_plot:
        _save_plot(X, Y, y_true, mean, std, sem_l(positions), args.res, Xtr, args.out)
        print(f"[plot] {args.out}")


def _save_plot(X, Y, y_true, mean, std, lengthscale, res, Xtr, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    extent = [X.min(), X.max(), Y.min(), Y.max()]
    panels = [
        ("Ground truth", y_true.reshape(res, res), "viridis"),
        ("SK-GP mean", mean.reshape(res, res), "viridis"),
        ("SK-GP uncertainty", std.reshape(res, res), "magma"),
        ("Semantic lengthscale", lengthscale.reshape(res, res), "cividis"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, (title, field, cmap) in zip(axes, panels):
        im = ax.imshow(field, origin="lower", extent=extent, cmap=cmap, aspect="equal")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[1].scatter(Xtr[:, 0], Xtr[:, 1], s=8, c="white", edgecolors="k", linewidths=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
