"""Entrypoint: train the SE(3) flow-matching backbone generator (Phase 2).

Thin wrapper around :func:`pdz_denovo.generative.train.main` so it can be run
without installing the package.

Examples:
    # 5-epoch smoke test on synthetic helices (no data / GPU needed):
    python scripts/train_generator.py --synthetic --epochs 5

    # Real run on downloaded backbones:
    python scripts/train_generator.py --data-dir data/processed --epochs 100
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pdz_denovo.generative.train import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
