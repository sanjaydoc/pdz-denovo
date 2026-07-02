"""Binding-affinity oracle for PSD-95 PDZ-domain binders.

PDZ domains recognize short C-terminal peptide motifs of their partners. PSD-95
PDZ domains are canonical **class I** recognizers of the motif:

        ... [S/T] - X - [hydrophobic]  -COOH
              (-2)  (-1)   (0, C-term)

with the extreme C-terminal residue buried in a hydrophobic pocket (commonly
V/I/L) and a Ser/Thr at the -2 position hydrogen-bonding a conserved His.

Because full protein-protein docking is heavy and `smina` targets small-molecule
ligands, the default oracle scores **C-terminal motif compatibility** with the
PSD-95 class I consensus -- a cheap, biologically grounded binding proxy. A
`smina` docking backend can be plugged in later behind the same interface.

Score is bounded ~[0, 1], higher = stronger predicted binding.
"""
from __future__ import annotations

import logging

from pdz_denovo.oracle.base import BaseOracle
from pdz_denovo.oracle.types import Candidate

LOGGER = logging.getLogger("pdz_denovo")

HYDROPHOBIC = set("VILMFWAY")
# Preferred residues at the C-terminal (P0) pocket for class I PSD-95 PDZ.
P0_PREF = {"V": 1.0, "I": 0.9, "L": 0.8, "F": 0.6, "A": 0.4, "M": 0.5}
# Preferred residues at the -2 position (class I hallmark: S/T).
P2_PREF = {"S": 1.0, "T": 1.0}


class BindingOracle(BaseOracle):
    name = "binding"

    def __init__(
        self,
        method: str = "motif",
        smina_exe: str | None = None,
        motif_window: int = 6,
        **kwargs,
    ) -> None:
        self.method = method
        self.smina_exe = smina_exe
        self.motif_window = motif_window
        self._extra = kwargs

    # --- motif-based proxy ---------------------------------------------------
    def _motif_score(self, seq: str) -> tuple[float, dict]:
        if len(seq) < 3:
            return 0.0, {"reason": "too_short"}

        p0 = seq[-1]  # C-terminal residue (position 0)
        p1 = seq[-2]  # position -1
        p2 = seq[-3]  # position -2

        # P0 pocket: hydrophobic C-terminus strongly preferred.
        p0_score = P0_PREF.get(p0, 0.05)
        # P2: class I hallmark Ser/Thr.
        p2_score = P2_PREF.get(p2, 0.1)
        # P1 tends to be permissive; mild bonus for non-proline, non-charged.
        p1_score = 0.6 if p1 not in set("PDEKR") else 0.3

        # Local hydrophobic context of the C-terminal window helps pocket burial.
        window = seq[-self.motif_window :]
        hydro_frac = sum(1 for a in window if a in HYDROPHOBIC) / len(window)
        context_score = _clamp01(hydro_frac / 0.6)  # ~60% hydrophobic saturates

        # Weighted combination emphasizing the two specificity determinants.
        score = (
            0.40 * p0_score
            + 0.30 * p2_score
            + 0.10 * p1_score
            + 0.20 * context_score
        )
        detail = {
            "P0": p0,
            "P-1": p1,
            "P-2": p2,
            "p0_score": round(p0_score, 3),
            "p2_score": round(p2_score, 3),
            "context_hydro_frac": round(hydro_frac, 3),
        }
        return float(_clamp01(score)), detail

    # --- smina backend (stub for future) -------------------------------------
    def _smina_score(self, candidate: Candidate) -> tuple[float, dict]:
        import os

        if not self.smina_exe or not os.path.exists(self.smina_exe):
            LOGGER.warning(
                "smina executable not found (%s); falling back to motif proxy.",
                self.smina_exe,
            )
            return self._motif_score(candidate.sequence)
        # Full docking integration would go here (prepare receptor/ligand,
        # run smina, parse affinity). Deferred until a Windows smina binary is
        # provided. For now, defer to the motif proxy to keep the loop working.
        return self._motif_score(candidate.sequence)

    # --- public API ----------------------------------------------------------
    def score(self, candidate: Candidate) -> float:
        if self.method == "smina":
            val, detail = self._smina_score(candidate)
        else:
            val, detail = self._motif_score(candidate.sequence)
        candidate.metadata.setdefault("binding_detail", {}).update(detail)
        return val


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))
