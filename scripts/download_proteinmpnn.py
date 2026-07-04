"""Clone ProteinMPNN (MIT) for the Phase 3 sequence-design stage.

ProteinMPNN has no official PyPI package, so we vendor it by cloning the
upstream repository (which ships its pretrained weights, including the Cα-only
models we use). The clone lands in ``third_party/ProteinMPNN`` which is
git-ignored — we depend on it, we don't re-distribute it.

Usage:
    python scripts/download_proteinmpnn.py
    python scripts/download_proteinmpnn.py --dest third_party/ProteinMPNN
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/dauparas/ProteinMPNN.git"


def main() -> int:
    parser = argparse.ArgumentParser(description="Clone ProteinMPNN for Phase 3.")
    parser.add_argument("--dest", default=str(ROOT / "third_party" / "ProteinMPNN"))
    parser.add_argument("--url", default=REPO_URL)
    args = parser.parse_args()

    dest = Path(args.dest)
    if (dest / "protein_mpnn_run.py").exists():
        print(f"ProteinMPNN already present at {dest}")
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {args.url} -> {dest} ...")
    try:
        subprocess.run(["git", "clone", "--depth", "1", args.url, str(dest)], check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Clone failed: {exc}", file=sys.stderr)
        return 1
    print("Done. ProteinMPNN (with Cα weights) is ready.")
    print(f"Point ProteinMPNNDesigner at: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
