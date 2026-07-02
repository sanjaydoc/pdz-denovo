"""Base oracle interface for the simulated wet-lab stack."""
from __future__ import annotations

from abc import ABC, abstractmethod

from pdz_denovo.oracle.types import Candidate


class BaseOracle(ABC):
    """Abstract single-objective scorer.

    Concrete oracles return a float where **higher is better**. If a raw metric
    is "lower is better" (e.g. docking energy), the concrete oracle is
    responsible for negating/normalizing it here.
    """

    name: str = "base"

    @abstractmethod
    def score(self, candidate: Candidate) -> float:
        """Return a scalar score (higher = better) for a single candidate."""

    def score_batch(self, candidates: list[Candidate]) -> list[float]:
        """Score a list of candidates. Override for true batching."""
        return [self.score(c) for c in candidates]
