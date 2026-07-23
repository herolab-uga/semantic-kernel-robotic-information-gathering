"""Scene-LLM QLoRA fine-tuning loop (Section III / Fig. 3).

Pipeline per procedurally generated world:

1. Build the environment, ground-truth RSSI field and sparse measurements.
2. Seed a lengthscale-ratio config from the current policy (or an LLM / defaults).
3. Draw ``K`` candidate parameterizations (policy samples + reward-guided
   perturbations), instantiate the Semantic Kernel for each, and reconstruct the
   field via GP inference.
4. Score candidates by (negative) reconstruction RMSE; the best is the positive
   example ``k*`` for the contrastive objective (Eq. 11).
5. Keep the best config as a replay example.

After all worlds, the highest-reward replay examples are distilled into the LoRA
adapter (:meth:`QLoRALengthscalePolicy.fine_tune`), which maximizes
``log pi_theta(k* | G_i)``.

If torch/gpytorch are unavailable, the AK-beat reward shaping is skipped and the
loop uses the pure negative-RMSE reward, so the search still runs end-to-end.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ..envs.environment import environment_from_scene
from ..envs.propagation import clip_dbm
from ..kernel.fields import SemanticLengthscale
from ..kernel.gate import build_gate_entities, resolve_gate_material_params
from ..kernel.gp import fit_predict_semantic_kernel
from ..kernel.hyperfield import SemanticHyperField
from ..scene_llm.config import (
    build_world_scaled_semantic_cfg,
    default_lengthscale_ratios,
    default_semantic_cfg,
    sanitize_lengthscale_ratios,
    sanitize_semantic_cfg,
)
from ..simulator.radio_world import world_stream
from .policy import QLoRALengthscalePolicy
from .records import ReplayExample, StepRecord
from .reward import (
    best_candidate_index,
    contrastive_cross_entropy,
    negative_rmse_reward,
    reward_from_rmse,
)

__all__ = ["perturb_lengthscale_ratios", "run_training", "main"]


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _sample_grid_bilinear(field: np.ndarray, points: np.ndarray, x_min, x_max, y_min, y_max) -> np.ndarray:
    field = np.asarray(field, dtype=np.float64)
    ny, nx = field.shape
    xs = (points[:, 0] - x_min) / max(x_max - x_min, 1e-12) * (nx - 1)
    ys = (points[:, 1] - y_min) / max(y_max - y_min, 1e-12) * (ny - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, nx - 1)
    y0 = np.clip(np.floor(ys).astype(int), 0, ny - 1)
    x1 = np.clip(x0 + 1, 0, nx - 1)
    y1 = np.clip(y0 + 1, 0, ny - 1)
    wx, wy = xs - x0, ys - y0
    return (
        (1 - wx) * (1 - wy) * field[y0, x0]
        + wx * (1 - wy) * field[y0, x1]
        + (1 - wx) * wy * field[y1, x0]
        + wx * wy * field[y1, x1]
    )


def perturb_lengthscale_ratios(base_cfg: Dict[str, Any], *, rng: np.random.Generator, scale: float) -> Dict[str, Any]:
    """Reward-guided log-normal perturbation of a lengthscale-ratio config."""
    scale = max(float(scale), 1e-3)
    out = {
        "tau_ratio": base_cfg["tau_ratio"] * float(np.exp(rng.normal(0.0, 0.22 * scale))),
        "beta": base_cfg["beta"] + float(rng.normal(0.0, 0.18 * scale)),
        "l_min_ratio": base_cfg["l_min_ratio"] * float(np.exp(rng.normal(0.0, 0.16 * scale))),
        "l_max_ratio": base_cfg["l_max_ratio"] * float(np.exp(rng.normal(0.0, 0.16 * scale))),
        "radius_ratio": base_cfg["radius_ratio"] * float(np.exp(rng.normal(0.0, 0.20 * scale))),
        "reasoning_summary": "Reward-guided perturbation.",
    }
    return sanitize_lengthscale_ratios(out, fallback=base_cfg)


def _evaluate_semantic(ctx: Dict[str, Any], ratio_cfg: Dict[str, Any]) -> float:
    """Reconstruct the field with a candidate config; return RMSE vs ground truth."""
    scene_scale = ctx["scene_scale"]
    sem_cfg = dict(ctx["base_sem_cfg"])
    sem_cfg["tau"] = ratio_cfg["tau_ratio"] * scene_scale
    sem_cfg["beta"] = ratio_cfg["beta"]
    sem_cfg["l_min"] = ratio_cfg["l_min_ratio"] * scene_scale
    sem_cfg["l_max"] = ratio_cfg["l_max_ratio"] * scene_scale
    sem_cfg["radius"] = ratio_cfg["radius_ratio"] * scene_scale
    if sem_cfg["l_max"] <= sem_cfg["l_min"]:
        sem_cfg["l_max"] = sem_cfg["l_min"] + 1e-6

    sem_l = SemanticLengthscale(
        ctx["env"], tau=sem_cfg["tau"], beta=sem_cfg["beta"], l_min=sem_cfg["l_min"],
        l_max=sem_cfg["l_max"], radius=sem_cfg["radius"],
    )
    hyperfield = SemanticHyperField(
        sem_l, K=sem_cfg["K"], temperature=sem_cfg["temperature"],
        center_mode=sem_cfg["center_mode"], priors=sem_cfg["priors"],
    )
    mean, _, _ = fit_predict_semantic_kernel(
        ctx["X_train"], ctx["y_train"], ctx["positions"], hyperfield=hyperfield,
        gate_entities=ctx["gate_entities"], gate_distance_scale=ctx["gate_distance_scale"],
        variance=sem_cfg["sem_variance"], jitter=sem_cfg["sem_jitter"], alpha=0.0,
        normalize_y=sem_cfg["normalize_y"],
        # This reward evaluation is part of *producing* the fine-tuned adapter,
        # so the un-tuned SK inference gate is bypassed here (see sk_gp._release).
        allow_untuned=True,
    )
    return _rmse(ctx["field"].reshape(-1), np.clip(mean, -50.0, 0.0))


def _prepare_context(world, complexity: str, n_samples: int, seed: int) -> Dict[str, Any]:
    scene_json = world.scene.to_json(hide_materials=False, hide_ap=False)
    env = environment_from_scene(scene_json)
    field = clip_dbm(np.asarray(world.true_map, dtype=np.float64))

    xx, yy = np.meshgrid(world.xs, world.ys)
    positions = np.column_stack([xx.reshape(-1), yy.reshape(-1)]).astype(np.float64)
    x_min, x_max = float(world.xs.min()), float(world.xs.max())
    y_min, y_max = float(world.ys.min()), float(world.ys.max())
    x_span = max(x_max - x_min, 1e-12)
    y_span = max(y_max - y_min, 1e-12)
    scene_scale = 0.75 * math.sqrt(x_span * y_span)
    gate_distance_scale = 0.50 * min(x_span, y_span)

    rng = np.random.default_rng(seed)
    X_train = np.column_stack([rng.uniform(x_min, x_max, n_samples), rng.uniform(y_min, y_max, n_samples)])
    y_train = _sample_grid_bilinear(field, X_train, x_min, x_max, y_min, y_max)

    base_sem_cfg = build_world_scaled_semantic_cfg(sanitize_semantic_cfg(default_semantic_cfg()), scene_scale=scene_scale)
    gate_params = resolve_gate_material_params(env, {})
    gate_entities = build_gate_entities(env, gate_params)
    n_entities = len(env.walls) + len(env.objects)
    return {
        "scene_json": scene_json,
        "env": env,
        "field": field,
        "positions": positions,
        "X_train": X_train,
        "y_train": y_train,
        "scene_scale": scene_scale,
        "gate_distance_scale": gate_distance_scale,
        "base_sem_cfg": base_sem_cfg,
        "gate_entities": gate_entities,
        "n_entities": n_entities,
        "complexity": complexity,
    }


def run_training(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = None
    if not args.skip_qlora:
        policy = QLoRALengthscalePolicy(
            base_model=args.base_model,
            trainable=True,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            max_seq_len=args.max_seq_len,
            learning_rate=args.qlora_lr,
            per_device_batch_size=args.qlora_batch_size,
            grad_accum_steps=args.qlora_grad_accum,
            train_epochs=args.qlora_epochs,
        )

    replay: List[ReplayExample] = []
    records: List[StepRecord] = []
    rng = np.random.default_rng(args.seed)

    stream = world_stream(args.n_simple, args.n_medium, args.n_complex, seed0=args.seed, res=args.res)
    for env_idx, world, complexity in stream:
        ctx = _prepare_context(world, complexity, n_samples=args.n_samples, seed=args.seed + 10_000 + env_idx)
        seed_cfg = default_lengthscale_ratios(ctx["scene_scale"])
        if policy is not None and policy.is_available:
            seed_cfg = policy.propose(
                prompt_text=json.dumps(world.scene.to_json(hide_materials=True), separators=(",", ":")),
                fallback_cfg=seed_cfg,
            )

        # Draw K candidates and evaluate reconstruction RMSE (the reward).
        candidates: List[Dict[str, Any]] = [seed_cfg]
        for _ in range(args.k_candidates - 1):
            candidates.append(perturb_lengthscale_ratios(seed_cfg, rng=rng, scale=0.6))
        rmses = [_evaluate_semantic(ctx, cfg) for cfg in candidates]
        rewards = [negative_rmse_reward(r, env_size=ctx["scene_scale"]) for r in rmses]

        k_star = best_candidate_index(rewards)
        loss = contrastive_cross_entropy(rewards, k_star, temperature=args.softmax_tau)
        best_cfg = candidates[k_star]

        replay.append(
            ReplayExample(
                env_idx=env_idx, complexity=complexity, reward=rewards[k_star],
                prompt_text=json.dumps(world.scene.to_json(hide_materials=True), separators=(",", ":")),
                answer_json=best_cfg,
            )
        )
        records.append(
            StepRecord(
                env_idx=env_idx, complexity=complexity, step=0, proposal_source="policy" if policy else "search",
                reward=rewards[k_star], rmse_candidate=rmses[k_star], rmse_sk_default=rmses[0],
                rmse_ak=float("nan"), best_rmse_so_far=rmses[k_star], beat_ak=False, proposal=best_cfg,
            )
        )
        print(f"[env {env_idx:03d} {complexity:7s}] best RMSE={rmses[k_star]:.3f}  contrastive_loss={loss:.3f}")

    # Distill the positive candidates into the LoRA adapter.
    if policy is not None and not args.skip_qlora:
        saved = policy.fine_tune(examples=replay, save_dir=out_dir / "saved_model", max_examples=args.max_replay_examples)
        print(f"[qlora] adapter saved: {saved}")

    (out_dir / "replay.jsonl").write_text(
        "\n".join(json.dumps({"env_idx": e.env_idx, "reward": e.reward, "answer": e.answer_json}) for e in replay)
    )
    return {"replay": replay, "records": records}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="QLoRA fine-tuning of the Scene-LLM lengthscale policy.")
    p.add_argument("--out-dir", default="runs/finetune")
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    p.add_argument("--n-simple", type=int, default=3)
    p.add_argument("--n-medium", type=int, default=3)
    p.add_argument("--n-complex", type=int, default=3)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--res", type=int, default=42)
    p.add_argument("--n-samples", type=int, default=60)
    p.add_argument("--k-candidates", type=int, default=8, help="K candidate parameterizations per environment (Eq. 11).")
    p.add_argument("--softmax-tau", type=float, default=1.0, help="Temperature tau in the candidate softmax (Eq. 11).")
    p.add_argument("--max-replay-examples", type=int, default=24)
    p.add_argument("--skip-qlora", action="store_true", help="Run the reward search without loading/training the LLM.")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--qlora-lr", type=float, default=2e-4)
    p.add_argument("--qlora-batch-size", type=int, default=2)
    p.add_argument("--qlora-grad-accum", type=int, default=8)
    p.add_argument("--qlora-epochs", type=float, default=5.0)
    return p


def main() -> None:
    run_training(_build_parser().parse_args())


if __name__ == "__main__":
    main()
