#!/usr/bin/env python3
"""Reconstruction experiment: SK vs baseline kernels (Fig. 5 a-d).

Runs the adaptive informative-sampling loop once per kernel on a scene and reports
final RMSE / uncertainty, optionally saving RMSE-vs-samples curves.

Usage
-----
    python experiments/run_reconstruction.py --scene data/scenes/house.json \
        --kernels sk rbf --n-samples 40 --noise-db 5 --save-plots

Add ``ak`` / ``dkl`` to ``--kernels`` once the ``baselines`` extra is installed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sk_gp.experiments_cli import reconstruction_main  # noqa: E402

if __name__ == "__main__":
    reconstruction_main()
