"""Shared datatypes for candidate designs and oracle scores."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

# Canonical 20 amino acids (single-letter).
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA_ALPHABET)


@dataclass
class Candidate:
    """A single designed protein candidate.

    Attributes:
        sequence: single-letter amino-acid sequence.
        id: stable identifier (e.g. hash or cycle-index string).
        origin: how it was produced ("seed", "flow", "mutation", ...).
        coords: optional (L, 3) or (L, N_atoms, 3) backbone coordinates.
        metadata: free-form provenance dict.
    """

    sequence: str
    id: Optional[str] = None
    origin: str = "unknown"
    coords: Optional[object] = None  # np.ndarray, kept generic to avoid hard dep
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sequence = self.sequence.strip().upper()
        if self.id is None:
            import hashlib

            self.id = hashlib.sha1(self.sequence.encode()).hexdigest()[:12]

    @property
    def length(self) -> int:
        return len(self.sequence)

    def is_valid(self) -> bool:
        """True if the sequence is non-empty and only canonical residues."""
        return len(self.sequence) > 0 and all(c in AA_SET for c in self.sequence)


@dataclass
class OracleScore:
    """Multi-objective scores for a candidate (all "higher is better").

    Raw sub-scores are stored plus a dict for any extra diagnostics.
    """

    stability: float = 0.0
    solubility: float = 0.0
    binding: float = 0.0
    details: dict = field(default_factory=dict)

    def as_vector(self) -> list[float]:
        """Return objectives in canonical order [stability, solubility, binding]."""
        return [self.stability, self.solubility, self.binding]

    def to_dict(self) -> dict:
        return asdict(self)
