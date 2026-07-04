"""NSGA-II multi-objective optimizer over protein sequences.

The "Learn -> propose next library" engine of the DBTL loop. Given a scored
population, NSGA-II (Deb et al., 2002) ranks candidates by Pareto front and
crowding distance, then produces a new library via tournament selection,
crossover, and mutation — the next batch to send to the (simulated) wet lab.

Sequence-space genetic operators are used because the design space is discrete;
the PDZ class-I motif is re-grafted after mutation so the domain constraint is
preserved. Pure-python (no torch), so the whole loop is testable on a laptop.
"""
from __future__ import annotations

import random

from pdz_denovo.oracle.types import AA_ALPHABET, Candidate
from pdz_denovo.optimize.pareto import crowding_distance, non_dominated_sort
from pdz_denovo.sequence.motif import graft_motif


class NSGA2Optimizer:
    """Propose the next library from a scored population via NSGA-II."""

    def __init__(
        self,
        mutation_rate: float = 0.05,
        crossover_rate: float = 0.7,
        alphabet: str = AA_ALPHABET,
        preserve_motif: bool = True,
        seed: int = 0,
    ) -> None:
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.alphabet = alphabet
        self.preserve_motif = preserve_motif
        self._rng = random.Random(seed)

    # --- ranking -------------------------------------------------------------
    def rank(self, scores) -> dict:
        """Return per-index ``(front_rank, crowding_distance)`` for a population."""
        fronts = non_dominated_sort(scores)
        info = {}
        for rank, front in enumerate(fronts):
            cd = crowding_distance(scores, front)
            for i in front:
                info[i] = (rank, cd[i])
        return info

    # --- genetic operators ---------------------------------------------------
    def _tournament(self, info: dict) -> int:
        """Binary tournament: lower front rank wins; ties broken by crowding."""
        a, b = self._rng.sample(list(info.keys()), 2) if len(info) > 1 else (
            list(info.keys())[0], list(info.keys())[0]
        )
        ra, ca = info[a]
        rb, cb = info[b]
        if ra != rb:
            return a if ra < rb else b
        return a if ca >= cb else b

    def _crossover(self, s1: str, s2: str) -> str:
        if len(s1) != len(s2) or self._rng.random() > self.crossover_rate:
            return s1
        # Uniform crossover.
        return "".join(self._rng.choice((c1, c2)) for c1, c2 in zip(s1, s2))

    def _mutate(self, seq: str) -> str:
        chars = list(seq)
        for i in range(len(chars)):
            if self._rng.random() < self.mutation_rate:
                chars[i] = self._rng.choice(self.alphabet)
        out = "".join(chars)
        return graft_motif(out) if self.preserve_motif else out

    # --- public API ----------------------------------------------------------
    def propose(self, population, scores, n: int) -> list[Candidate]:
        """Produce ``n`` offspring candidates from a scored population.

        Args:
            population: list of :class:`Candidate` evaluated last cycle.
            scores: list of objective vectors (maximization), aligned to
                ``population``.
            n: size of the next library to propose.

        Returns:
            ``n`` new :class:`Candidate` objects (origin ``"nsga2"``).
        """
        info = self.rank(scores)
        offspring: list[Candidate] = []
        for _ in range(n):
            pi = self._tournament(info)
            qi = self._tournament(info)
            child_seq = self._crossover(population[pi].sequence, population[qi].sequence)
            child_seq = self._mutate(child_seq)
            offspring.append(
                Candidate(
                    sequence=child_seq,
                    origin="nsga2",
                    metadata={"parents": [population[pi].id, population[qi].id]},
                )
            )
        return offspring
