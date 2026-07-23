#!/usr/bin/env python3
"""Noise robustness sweep (Fig. 5 e-f).

Evaluates each kernel at the two paper noise levels (N1 = 5 dB, N2 = 10 dB) and
prints a final-RMSE table plus the average RMSE increase from N1 to N2 -- the
robustness-to-noise metric the paper reports SK improving on.

    python experiments/run_noise_sweep.py --scene data/scenes/house.json --kernels sk rbf
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sk_gp.experiments_cli import reconstruction_main  # noqa: E402
from sk_gp.simulator.radio_world import NOISE_LEVELS_DB  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Noise robustness sweep over N1/N2.")
    parser.add_argument("--scene", default="data/scenes/house.json")
    parser.add_argument("--kernels", nargs="+", default=["sk", "rbf"])
    parser.add_argument("--n-samples", type=int, default=40)
    parser.add_argument("--res", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="runs/noise_sweep")
    args = parser.parse_args()

    table = {}
    for level_name, noise_db in NOISE_LEVELS_DB.items():
        print(f"\n=== Noise level {level_name} ({noise_db} dB) ===")
        summary = reconstruction_main(
            [
                "--scene", args.scene,
                "--kernels", *args.kernels,
                "--n-samples", str(args.n_samples),
                "--res", str(args.res),
                "--noise-db", str(noise_db),
                "--seed", str(args.seed),
                "--out-dir", f"{args.out_dir}/{level_name}",
            ]
        )
        table[level_name] = {k: v["final_rmse"] for k, v in summary.items()}

    print("\n=== Final RMSE table ===")
    kernels = sorted({k for level in table.values() for k in level})
    print("level " + "  ".join(f"{k:>8s}" for k in kernels))
    for level_name, row in table.items():
        print(f"{level_name:5s} " + "  ".join(f"{row.get(k, float('nan')):8.3f}" for k in kernels))

    if "N1" in table and "N2" in table:
        print("\nRMSE increase N1 -> N2 (lower = more robust):")
        for k in kernels:
            if k in table["N1"] and k in table["N2"]:
                print(f"  {k:>4s}: {table['N2'][k] - table['N1'][k]:+.3f}")


if __name__ == "__main__":
    main()
