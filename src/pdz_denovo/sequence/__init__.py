"""Sequence design (inverse folding) — assign sequences to backbones (Phase 3).

Turns Phase 2 Cα backbones into candidate sequences with the PDZ class-I motif
enforced, emitted as oracle-ready :class:`~pdz_denovo.oracle.types.Candidate`
objects. Torch-free at import time (the heavy model runs as a subprocess), so
this package imports on a bare machine.
"""
from pdz_denovo.sequence.motif import (
    describe_motif,
    graft_motif,
    motif_satisfied,
)
from pdz_denovo.sequence.proteinmpnn import (
    FallbackDesigner,
    ProteinMPNNDesigner,
    parse_mpnn_fasta,
    pdb_ca_length,
)

__all__ = [
    "describe_motif",
    "graft_motif",
    "motif_satisfied",
    "FallbackDesigner",
    "ProteinMPNNDesigner",
    "parse_mpnn_fasta",
    "pdb_ca_length",
]
