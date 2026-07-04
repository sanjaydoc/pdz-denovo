"""End-to-end validation: Design -> Build -> Test (Phase 4).

Runs the full front half of the DBTL loop and reports self-consistency:

    trained generator  ->  sample Cα backbones
                       ->  ProteinMPNN sequences (+ PDZ motif)
                       ->  ESMFold refold  ->  scRMSD / pLDDT

This is the "money" pipeline that ties Phases 2–4 together and produces the
credibility metric reviewers look for.

Examples:
    # API folding (no local VRAM), a couple of backbones, a few seqs each:
    python scripts/validate_designs.py --n-backbones 2 --n-seqs 4

    # Use a specific checkpoint / local ProteinMPNN clone:
    python scripts/validate_designs.py \
        --checkpoint outputs/generator/model.pt \
        --mpnn-dir third_party/ProteinMPNN --fold-backend api
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pdz_denovo.utils.common import setup_logging  # noqa: E402

LOGGER = setup_logging()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate designs via self-consistency.")
    p.add_argument("--checkpoint", default="outputs/generator/model.pt")
    p.add_argument("--mpnn-dir", default="third_party/ProteinMPNN")
    p.add_argument("--n-backbones", type=int, default=2)
    p.add_argument("--length", type=int, default=64)
    p.add_argument("--n-seqs", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--fold-backend", choices=["api", "local"], default="api")
    p.add_argument("--out", default="outputs/validation")
    p.add_argument("--use-fallback-designer", action="store_true",
                   help="Use the random fallback designer instead of ProteinMPNN.")
    return p.parse_args()


def main() -> int:
    import torch
    from omegaconf import OmegaConf

    from pdz_denovo.fold import ESMFoldBackend, summarize, validate_candidates
    from pdz_denovo.generative.flow import build_flow_model
    from pdz_denovo.generative.sample import coords_to_pdb
    from pdz_denovo.sequence import FallbackDesigner, ProteinMPNNDesigner

    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the trained generator and sample backbones.
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = build_flow_model(OmegaConf.create(ckpt["flow_cfg"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    LOGGER.info("Sampling %d backbones (L=%d) ...", args.n_backbones, args.length)
    coords = model.sample(n_samples=args.n_backbones, length=args.length, n_steps=args.n_steps)

    # 2. Sequence design per backbone.
    if args.use_fallback_designer:
        designer = FallbackDesigner(seed=0)
        LOGGER.warning("Using FallbackDesigner (random, structure-agnostic).")
    else:
        designer = ProteinMPNNDesigner(repo_dir=args.mpnn_dir)

    fold_backend = ESMFoldBackend(method=args.fold_backend)
    all_results = []
    for i in range(args.n_backbones):
        bb_id = f"design_{i:03d}"
        pdb_path = coords_to_pdb(coords[i].cpu(), out_dir / f"{bb_id}.pdb")
        cands = designer.design(pdb_path=pdb_path, n_seqs=args.n_seqs, backbone_id=bb_id)

        # 3. Refold + self-consistency.
        results = validate_candidates(coords[i].cpu().numpy(), cands, fold_backend, backbone_id=bb_id)
        all_results.extend(results)
        best = results[0] if results else {}
        LOGGER.info("%s: best scRMSD=%s pLDDT=%s", bb_id, best.get("scRMSD"), best.get("plddt"))

    summary = summarize(all_results)
    (out_dir / "validation.json").write_text(
        json.dumps({"summary": summary, "results": all_results}, indent=2)
    )
    LOGGER.info("Summary: %s", summary)
    LOGGER.info("Wrote %s", out_dir / "validation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
