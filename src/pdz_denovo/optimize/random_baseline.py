"""Random-search baseline optimizer.

The control the DBTL loop is measured against: it ignores the oracle scores and
proposes a fresh random library (with the PDZ motif grafted) every cycle. If
NSGA-II / qNEHVI don't beat this on hypervolume, the "optimization" isn't doing
anything — so the benchmark in ``scripts/benchmark_optimizers.py`` reports all
three side by side.
"""
from __future__ import annotations

import random

from pdz_denovo.oracle.types import AA_ALPHABET, Candidate
from pdz_denovo.sequence.motif import graft_motif


class RandomOptimizer:
    """Propose a fresh random library each cycle (no selection pressure)."""

    def __init__(self, seed: int = 0, preserve_motif: bool = True, alphabet: str = AA_ALPHABET) -> None:
        self._rng = random.Random(seed)
        self.preserve_motif = preserve_motif
        self.alphabet = alphabet

    def propose(self, population, scores, n: int) -> list[Candidate]:
        length = population[0].length if population else 64
        out = []
        for _ in range(n):
            seq = "".join(self._rng.choice(self.alphabet) for _ in range(length))
            if self.preserve_motif:
                seq = graft_motif(seq)
            out.append(Candidate(sequence=seq, origin="random"))
        return out
