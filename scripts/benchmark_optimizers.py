"""Benchmark the DBTL optimizers against a random-search baseline (Phase 6).

Runs the same closed loop (same seed library, same oracle) with each optimizer
and compares **hypervolume trajectories**. If NSGA-II / qNEHVI don't beat random
search, the optimization adds nothing — so this is the experiment that proves the
loop works.

Outputs ``outputs/benchmark/benchmark.json`` and a comparison PNG.

Examples:
    python scripts/benchmark_optimizers.py --n-cycles 6 --library-size 24 --n-seed 12
    python scripts/benchmark_optimizers.py --optimizers nsga2 random
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pdz_denovo.utils.common import set_seed, setup_logging  # noqa: E402

LOGGER = setup_logging()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark DBTL optimizers.")
    p.add_argument("--n-cycles", type=int, default=6)
    p.add_argument("--library-size", type=int, default=24)
    p.add_argument("--n-seed", type=int, default=12)
    p.add_argument("--length", type=int, default=64)
    p.add_argument("--esm-model", default="esm2_t12_35M_UR50D")
    p.add_argument("--optimizers", nargs="+", default=["random", "nsga2", "qnehvi"])
    p.add_argument("--out", default="outputs/benchmark")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_optimizer(name: str, seed: int, esm_model: str):
    if name == "random":
        from pdz_denovo.optimize import RandomOptimizer

        return RandomOptimizer(seed=seed)
    if name == "nsga2":
        from pdz_denovo.optimize import NSGA2Optimizer

        return NSGA2Optimizer(seed=seed)
    if name == "qnehvi":
        from pdz_denovo.optimize.features import ESMFeaturizer
        from pdz_denovo.optimize.qnehvi import QNEHVIProposer

        return QNEHVIProposer(
            featurizer=ESMFeaturizer(esm_model=esm_model), ref_point=[-8.0, 0.0, 0.0], seed=seed
        )
    raise ValueError(f"Unknown optimizer '{name}'")


def main() -> int:
    from pdz_denovo.loop import DBTLoop
    from pdz_denovo.oracle import (
        BindingOracle,
        OracleStack,
        SolubilityOracle,
        StabilityOracle,
    )
    from pdz_denovo.sequence import FallbackDesigner

    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # One shared oracle (cache makes repeated candidates cheap across methods).
    stack = OracleStack(
        stability=StabilityOracle(esm_model=args.esm_model, normalize=False),
        solubility=SolubilityOracle(),
        binding=BindingOracle(method="motif"),
        cache=True,
    )

    def scorer(cands):
        return [s.as_vector() for s in stack.score_batch(cands)]

    # Fixed reference and identical seed library across methods for fairness.
    ref_point = [-8.0, 0.0, 0.0]
    results = {}
    for name in args.optimizers:
        LOGGER.info("=== Benchmarking optimizer: %s ===", name)
        seed_designer = FallbackDesigner(seed=args.seed)  # identical seeds per method
        loop = DBTLoop(
            scorer=scorer,
            optimizer=make_optimizer(name, args.seed, args.esm_model),
            seed_fn=lambda n: seed_designer.design(length=args.length, n_seqs=n),
            ref_point=list(ref_point),
        )
        res = loop.run(
            n_cycles=args.n_cycles, library_size=args.library_size, n_seed=args.n_seed
        )
        hv = [h["hypervolume"] for h in res["history"]]
        results[name] = {"hypervolume": hv, "final_pareto": len(res["pareto"])}
        LOGGER.info("%s | HV trajectory: %s", name, [round(v, 4) for v in hv])

    (out_dir / "benchmark.json").write_text(json.dumps(results, indent=2))

    # Plot (matplotlib is optional; skip gracefully if unavailable).
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name, r in results.items():
            ax.plot(range(len(r["hypervolume"])), r["hypervolume"], marker="o", label=name)
        ax.set_xlabel("DBTL cycle")
        ax.set_ylabel("Hypervolume")
        ax.set_title("Optimizer comparison — hypervolume vs cycle")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "benchmark.png", dpi=130)
        LOGGER.info("Wrote %s", out_dir / "benchmark.png")
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Plot skipped (%s).", exc)

    LOGGER.info("Wrote %s", out_dir / "benchmark.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
