"""Solubility / aggregation-propensity oracle (sequence-based, lightweight).

Combines three interpretable biophysical signals commonly associated with
soluble, well-behaved proteins:

1. Low mean hydrophobicity (Kyte-Doolittle) => more soluble.
2. Adequate net charge magnitude at pH 7 => electrostatic repulsion reduces
   aggregation.
3. Few long hydrophobic / aggregation-prone stretches.

The output is a bounded score in roughly [0, 1] where higher = more soluble.
This is intentionally a *proxy* oracle (noisy, cheap), mirroring how a real
DBTL loop starts before high-quality assays exist.
"""
from __future__ import annotations

from pdz_denovo.oracle.base import BaseOracle
from pdz_denovo.oracle.types import Candidate

# Kyte-Doolittle hydropathy index.
KD_HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Approximate charge at pH 7.
POS_RESIDUES = {"K", "R", "H"}  # H partial; treated as mildly positive
NEG_RESIDUES = {"D", "E"}


class SolubilityOracle(BaseOracle):
    name = "solubility"

    def __init__(
        self,
        hydrophobicity_weight: float = 1.0,
        charge_weight: float = 0.5,
        aggregation_weight: float = 1.0,
        agg_window: int = 5,
        agg_threshold: float = 2.0,
    ) -> None:
        self.w_hydro = hydrophobicity_weight
        self.w_charge = charge_weight
        self.w_agg = aggregation_weight
        self.agg_window = agg_window
        self.agg_threshold = agg_threshold

    # --- component signals ---------------------------------------------------
    def mean_hydropathy(self, seq: str) -> float:
        vals = [KD_HYDROPATHY.get(a, 0.0) for a in seq]
        return sum(vals) / max(len(vals), 1)

    def net_charge(self, seq: str) -> float:
        pos = sum(1.0 for a in seq if a in POS_RESIDUES)
        neg = sum(1.0 for a in seq if a in NEG_RESIDUES)
        return pos - neg

    def aggregation_fraction(self, seq: str) -> float:
        """Fraction of windows whose mean hydropathy exceeds the threshold."""
        if len(seq) < self.agg_window:
            return 1.0 if self.mean_hydropathy(seq) > self.agg_threshold else 0.0
        n_windows = len(seq) - self.agg_window + 1
        hot = 0
        for i in range(n_windows):
            window = seq[i : i + self.agg_window]
            if self.mean_hydropathy(window) > self.agg_threshold:
                hot += 1
        return hot / n_windows

    # --- scoring -------------------------------------------------------------
    def score(self, candidate: Candidate) -> float:
        seq = candidate.sequence
        if not seq:
            return 0.0

        # 1. Hydrophobicity term: map mean KD (~[-4.5, 4.5]) to [0,1], lower=better.
        mh = self.mean_hydropathy(seq)
        hydro_term = _clamp01(0.5 - (mh / 9.0))  # mh=0 -> 0.5; very hydrophilic -> ~1

        # 2. Charge term: reward |net charge| per length up to a saturation point.
        charge_density = abs(self.net_charge(seq)) / len(seq)
        charge_term = _clamp01(charge_density / 0.15)  # ~15% charged residues saturates

        # 3. Aggregation term: fewer hydrophobic stretches = better.
        agg_term = 1.0 - self.aggregation_fraction(seq)

        total_w = self.w_hydro + self.w_charge + self.w_agg
        score = (
            self.w_hydro * hydro_term
            + self.w_charge * charge_term
            + self.w_agg * agg_term
        ) / total_w

        candidate.metadata.setdefault("solubility_detail", {}).update(
            {
                "mean_hydropathy": round(mh, 3),
                "net_charge": self.net_charge(seq),
                "aggregation_fraction": round(self.aggregation_fraction(seq), 3),
            }
        )
        return float(_clamp01(score))


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))
