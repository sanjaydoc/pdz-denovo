"""DBTL loop orchestrator — the closed-loop optimization backbone.

Ties the whole platform together:

    seed library                     (Design)
      -> oracle scores               (Test, in silico)
      -> optimizer proposes next     (Learn)
      -> repeat                      (next cycle)

The loop is deliberately **component-agnostic**: it takes a ``scorer`` (any
callable ``list[Candidate] -> list[objective-vector]``), an ``optimizer`` (any
object with ``propose(population, scores, n) -> list[Candidate]``), and a
``seed_fn``. That decoupling is the point — swap the oracle/target and the same
backbone drives a different campaign (the "multiple verticals" the role targets),
and it lets the loop be unit-tested with a trivial scorer, no torch required.

Each cycle records Pareto-front size and **hypervolume** (the standard
multi-objective progress metric) over the cumulative archive of everything
evaluated so far.
"""
from __future__ import annotations

import logging

from pdz_denovo.optimize.pareto import hypervolume, pareto_front_indices

LOGGER = logging.getLogger("pdz_denovo")


class DBTLoop:
    def __init__(self, scorer, optimizer, seed_fn, tracker=None, ref_point=None) -> None:
        """
        Args:
            scorer: ``list[Candidate] -> list[list[float]]`` (maximization).
            optimizer: object with ``propose(population, scores, n)``.
            seed_fn: ``int -> list[Candidate]`` producing the initial library.
            tracker: optional object with ``log_cycle(dict)`` / ``finish()``.
            ref_point: hypervolume reference (lower bounds). If None, derived from
                the first cycle's worst scores and then held fixed.
        """
        self.scorer = scorer
        self.optimizer = optimizer
        self.seed_fn = seed_fn
        self.tracker = tracker
        self.ref_point = ref_point

    def _ensure_ref(self, scores) -> list[float]:
        if self.ref_point is not None:
            return self.ref_point
        # Fixed reference just below the worst observed point, so hypervolume is
        # comparable across cycles.
        m = len(scores[0])
        self.ref_point = [min(s[k] for s in scores) - 0.05 for k in range(m)]
        return self.ref_point

    def run(self, n_cycles: int = 5, library_size: int = 32, n_seed: int = 16) -> dict:
        """Run the DBTL loop and return history, archive, and final Pareto set."""
        population = self.seed_fn(n_seed)
        archive: list[tuple] = []  # (Candidate, score_vector), unique by sequence
        seen: set = set()  # sequences already in the archive
        history: list[dict] = []

        for cycle in range(n_cycles):
            scores = self.scorer(population)
            for cand, vec in zip(population, scores):
                cand.metadata["objectives"] = list(vec)
                # The same sequence can be re-proposed across cycles (elites,
                # repeated mutants); keep the archive unique so the Pareto front
                # and metrics never double-count a design.
                if cand.sequence in seen:
                    continue
                seen.add(cand.sequence)
                archive.append((cand, list(vec)))

            arch_scores = [v for _, v in archive]
            ref = self._ensure_ref(arch_scores)
            pf_idx = pareto_front_indices(arch_scores)
            hv = hypervolume(arch_scores, ref, maximize=True)
            best_per_obj = [max(s[k] for s in arch_scores) for k in range(len(ref))]

            metrics = {
                "cycle": cycle,
                "library_size": len(population),
                "archive_size": len(archive),
                "n_pareto": len(pf_idx),
                "hypervolume": round(hv, 6),
                "best_per_objective": [round(b, 4) for b in best_per_obj],
            }
            history.append(metrics)
            LOGGER.info(
                "cycle %d | archive=%d | pareto=%d | HV=%.4f | best=%s",
                cycle, len(archive), len(pf_idx), hv, metrics["best_per_objective"],
            )
            if self.tracker is not None:
                self.tracker.log_cycle(metrics)

            if cycle < n_cycles - 1:
                population = self.optimizer.propose(population, scores, library_size)

        final_scores = [v for _, v in archive]
        pareto = [archive[i] for i in pareto_front_indices(final_scores)]
        if self.tracker is not None:
            self.tracker.finish({"history": history})
        return {"history": history, "archive": archive, "pareto": pareto, "ref_point": self.ref_point}
