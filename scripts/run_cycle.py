"""Run the closed-loop DBTL optimization (Phase 5) — the platform end-to-end.

    seed library -> oracle scores -> optimizer proposes next library -> repeat

Objectives (all maximized): ESM-2 stability, solubility, PDZ binding. Progress
is tracked as Pareto-front size and hypervolume per cycle (MLflow + JSON).

Examples:
    # Fast: random+motif seed, NSGA-II, 3-objective oracle:
    python scripts/run_cycle.py --n-cycles 5 --library-size 32 --n-seed 16

    # Bayesian optimization instead of the genetic algorithm:
    python scripts/run_cycle.py --optimizer qnehvi

    # Seed from the trained generator + ProteinMPNN (full Design stage):
    python scripts/run_cycle.py --seed-source generative \
        --checkpoint outputs/generator/model.pt --mpnn-dir third_party/ProteinMPNN
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pdz_denovo.utils.common import set_seed, setup_logging  # noqa: E402

LOGGER = setup_logging()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the DBTL closed loop.")
    p.add_argument("--n-cycles", type=int, default=5)
    p.add_argument("--library-size", type=int, default=32)
    p.add_argument("--n-seed", type=int, default=16)
    p.add_argument("--length", type=int, default=64)
    p.add_argument("--optimizer", choices=["nsga2", "qnehvi"], default="nsga2")
    p.add_argument("--seed-source", choices=["random", "generative"], default="random")
    p.add_argument("--checkpoint", default="outputs/generator/model.pt")
    p.add_argument("--mpnn-dir", default="third_party/ProteinMPNN")
    p.add_argument("--esm-model", default="esm2_t12_35M_UR50D")
    p.add_argument("--out", default="outputs/dbtl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-mlflow", action="store_true")
    return p.parse_args()


def build_scorer(esm_model: str):
    from pdz_denovo.oracle import (
        BindingOracle,
        OracleStack,
        SolubilityOracle,
        StabilityOracle,
    )

    stack = OracleStack(
        stability=StabilityOracle(esm_model=esm_model, normalize=False),
        solubility=SolubilityOracle(),
        binding=BindingOracle(method="motif"),
        cache=True,
    )

    def scorer(cands):
        return [s.as_vector() for s in stack.score_batch(cands)]

    return scorer


def build_seed_fn(args):
    from pdz_denovo.sequence import FallbackDesigner

    if args.seed_source == "random":
        designer = FallbackDesigner(seed=args.seed)
        return lambda n: designer.design(length=args.length, n_seqs=n)

    # Generative seed: sample backbones then ProteinMPNN one sequence each.
    import torch
    from omegaconf import OmegaConf

    from pdz_denovo.generative.flow import build_flow_model
    from pdz_denovo.generative.sample import coords_to_pdb
    from pdz_denovo.sequence import FallbackDesigner, ProteinMPNNDesigner

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = build_flow_model(OmegaConf.create(ckpt["flow_cfg"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    designer = ProteinMPNNDesigner(repo_dir=args.mpnn_dir)
    tmp_dir = Path(args.out) / "seed_backbones"

    def seed_fn(n):
        coords = model.sample(n_samples=n, length=args.length, n_steps=100)
        seeds = []
        for i in range(n):
            c = coords[i]
            # A small ODE sampler occasionally emits a degenerate backbone
            # (non-finite or collapsed coords) that crashes ProteinMPNN's native
            # code. Skip those, and skip any single design that still fails, so
            # one bad backbone never aborts the whole campaign.
            if not torch.isfinite(c).all() or float(c.std()) < 1e-3:
                LOGGER.warning("Skipping degenerate backbone seed_%03d.", i)
                continue
            pdb = coords_to_pdb(c.cpu(), tmp_dir / f"seed_{i:03d}.pdb")
            try:
                seeds.extend(designer.design(pdb_path=pdb, n_seqs=1, backbone_id=f"seed_{i:03d}"))
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("ProteinMPNN failed on seed_%03d (%s); skipping.", i, exc)
        if not seeds:
            LOGGER.warning("No generative seeds succeeded; falling back to random seeds.")
            seeds = FallbackDesigner(seed=args.seed).design(length=args.length, n_seqs=n)
        return seeds

    return seed_fn


def build_optimizer(args):
    if args.optimizer == "nsga2":
        from pdz_denovo.optimize import NSGA2Optimizer

        return NSGA2Optimizer(mutation_rate=0.05, crossover_rate=0.7, seed=args.seed)

    from pdz_denovo.optimize.features import ESMFeaturizer
    from pdz_denovo.optimize.qnehvi import QNEHVIProposer

    featurizer = ESMFeaturizer(esm_model=args.esm_model)
    # Reference point (lower bounds) for [stability, solubility, binding].
    return QNEHVIProposer(featurizer=featurizer, ref_point=[-8.0, 0.0, 0.0], seed=args.seed)


def main() -> int:
    from pdz_denovo.loop import DBTLoop
    from pdz_denovo.tracking import RunTracker

    args = parse_args()
    set_seed(args.seed)

    scorer = build_scorer(args.esm_model)
    seed_fn = build_seed_fn(args)
    optimizer = build_optimizer(args)
    tracker = RunTracker(
        out_dir=args.out,
        use_mlflow=not args.no_mlflow,
        params={
            "optimizer": args.optimizer,
            "seed_source": args.seed_source,
            "n_cycles": args.n_cycles,
            "library_size": args.library_size,
            "n_seed": args.n_seed,
            "length": args.length,
        },
    )

    loop = DBTLoop(scorer=scorer, optimizer=optimizer, seed_fn=seed_fn, tracker=tracker)
    result = loop.run(
        n_cycles=args.n_cycles, library_size=args.library_size, n_seed=args.n_seed
    )

    pareto = result["pareto"]
    # Persist the Pareto set (id, sequence, objectives) for the dashboard.
    import json

    pareto_records = [
        {
            "id": cand.id,
            "sequence": cand.sequence,
            "origin": cand.origin,
            "stability": vec[0],
            "solubility": vec[1],
            "binding": vec[2],
        }
        for cand, vec in pareto
    ]
    (Path(args.out) / "pareto.json").write_text(json.dumps(pareto_records, indent=2))
    LOGGER.info("Done. Final Pareto set: %d designs.", len(pareto))
    for cand, vec in sorted(pareto, key=lambda cv: -cv[1][2])[:10]:
        LOGGER.info(
            "  %s | stab=%.3f sol=%.3f bind=%.3f | %s",
            cand.id, vec[0], vec[1], vec[2], cand.sequence,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
