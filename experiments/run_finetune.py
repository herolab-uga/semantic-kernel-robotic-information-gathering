#!/usr/bin/env python3
"""Scene-LLM QLoRA fine-tuning entry point (Section III / Fig. 3).

Thin wrapper over :func:`sk_gp.finetune.train.main`.  Run the reward search without
the LLM (no GPU / HF stack needed) via ``--skip-qlora``:

    python experiments/run_finetune.py --skip-qlora --n-simple 2 --n-medium 1 --n-complex 1

Full QLoRA fine-tuning (requires the ``finetune`` extra and a GPU):

    python experiments/run_finetune.py --base-model Qwen/Qwen3-8B \
        --n-simple 60 --n-medium 60 --n-complex 80 --k-candidates 8
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sk_gp.finetune.train import main  # noqa: E402

if __name__ == "__main__":
    main()
