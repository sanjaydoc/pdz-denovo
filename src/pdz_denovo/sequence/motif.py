"""PSD-95 PDZ class-I C-terminal motif — the encoded domain prior.

PSD-95 PDZ domains are canonical **class I** recognizers of the C-terminal
motif::

        ... [S/T] - X - Φ  -COOH
             (-2)  (-1)  (0)

where position 0 (the extreme C-terminus) is buried in a hydrophobic pocket
(commonly V/I/L) and position -2 is a Ser/Thr that hydrogen-bonds a conserved
histidine. Our design strategy is **C-terminal presentation**: the miniprotein
engages the PDZ groove via its own C-terminal tail, so a valid binder must end
in this motif.

This module encodes that prior as a small, dependency-free utility used by the
sequence-design stage: it can *check* whether a sequence satisfies the motif and
*graft* the motif onto a designed sequence so the constraint is guaranteed. The
motif definition is imported from :mod:`pdz_denovo.oracle.binding` so the
designer and the binding oracle share a single source of truth.
"""
from __future__ import annotations

from pdz_denovo.oracle.binding import HYDROPHOBIC, P2_PREF

# Default residues used when grafting a canonical strong class-I binder (…TSV).
DEFAULT_P0 = "V"  # extreme C-terminus: buried hydrophobic
DEFAULT_P2 = "T"  # -2 position: Ser/Thr hallmark

# Residues acceptable at each specificity-determining position.
P0_ALLOWED = set(HYDROPHOBIC)      # position 0 (C-terminus)
P2_ALLOWED = set(P2_PREF.keys())   # position -2 (S/T)


def motif_satisfied(seq: str) -> bool:
    """True if ``seq`` already ends in a valid class-I motif.

    Requires length ≥ 3, a hydrophobic C-terminus (P0) and Ser/Thr at P-2.
    """
    if len(seq) < 3:
        return False
    p0 = seq[-1]
    p2 = seq[-3]
    return p0 in P0_ALLOWED and p2 in P2_ALLOWED


def graft_motif(seq: str, p0: str = DEFAULT_P0, p2: str = DEFAULT_P2) -> str:
    """Return ``seq`` with the class-I motif enforced at its C-terminus.

    Only the specificity-determining positions are changed, and only if they do
    not already satisfy the motif — so a design that already ends correctly is
    left untouched. Position -1 (permissive) is never modified.

    Args:
        seq: designed amino-acid sequence.
        p0: residue to place at the C-terminus if it is not already hydrophobic.
        p2: residue to place at -2 if it is not already Ser/Thr.

    Returns:
        The (possibly) motif-corrected sequence, same length as the input.
    """
    if len(seq) < 3:
        return seq
    chars = list(seq)
    if chars[-1] not in P0_ALLOWED:
        chars[-1] = p0
    if chars[-3] not in P2_ALLOWED:
        chars[-3] = p2
    return "".join(chars)


def describe_motif(seq: str) -> dict:
    """Return a small diagnostic dict about the C-terminal motif of ``seq``."""
    if len(seq) < 3:
        return {"satisfied": False, "reason": "too_short"}
    return {
        "satisfied": motif_satisfied(seq),
        "P0": seq[-1],
        "P-1": seq[-2],
        "P-2": seq[-3],
    }
