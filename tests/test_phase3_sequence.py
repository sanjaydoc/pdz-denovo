"""Phase 3 tests for sequence design (inverse folding).

These are dependency-free (no torch, no ProteinMPNN weights): they cover the
motif prior, the FASTA parser, the CLI command construction, and the fallback
designer / Candidate integration — i.e. every part of the stage *except* the
actual ProteinMPNN forward pass, which is validated on a machine that has the
cloned repo.
"""
from __future__ import annotations

from pathlib import Path

from pdz_denovo.oracle.types import AA_SET
from pdz_denovo.sequence import (
    FallbackDesigner,
    ProteinMPNNDesigner,
    describe_motif,
    graft_motif,
    motif_satisfied,
    parse_mpnn_fasta,
    pdb_ca_length,
)

# --- motif prior -------------------------------------------------------------


def test_motif_satisfied_recognises_class1():
    assert motif_satisfied("GLGFKESTSV")  # ...T-S-V : S/T at -2, hydrophobic V
    assert not motif_satisfied("GLGFKESDDD")  # acidic, non-hydrophobic C-term
    assert not motif_satisfied("AV")  # too short


def test_graft_enforces_and_preserves_length():
    bad = "AAAAAAAAAA"  # ends A..A, P-2 = A (not S/T), P0 = A (hydrophobic-ok)
    grafted = graft_motif(bad)
    assert len(grafted) == len(bad)
    assert motif_satisfied(grafted)
    # P-2 must have become S/T; P0 was already hydrophobic (A) so may be kept.
    assert grafted[-3] in {"S", "T"}


def test_graft_leaves_valid_sequence_untouched():
    good = "MKTAYIAKTSV"  # already ...T-S-V
    assert graft_motif(good) == good


def test_describe_motif_fields():
    d = describe_motif("MKTAYITSV")
    assert d["satisfied"] is True
    assert d["P0"] == "V" and d["P-2"] == "T"


# --- FASTA parsing -----------------------------------------------------------

_SAMPLE_FA = """\
>design_000, score=1.0, global_score=1.0, model_name=v_48_002
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ
>T=0.1, sample=1, score=0.85, global_score=0.85
GLTAYIAKQRQISFVKSHFSRELEERLGLIEVQ
>T=0.1, sample=2, score=0.88, global_score=0.88
MKSAYIAKQRQISFVKSHFSRQLDERLGLIEVQ
"""


def test_parse_fasta_skips_native_by_default():
    recs = parse_mpnn_fasta(_SAMPLE_FA)
    assert len(recs) == 2  # native (first) dropped
    assert recs[0][1].startswith("GLTAYI")
    assert all(len(seq) == 33 for _, seq in recs)


def test_parse_fasta_can_keep_native():
    recs = parse_mpnn_fasta(_SAMPLE_FA, skip_first=False)
    assert len(recs) == 3


# --- CLI command construction ------------------------------------------------


def test_build_command_has_ca_only_and_core_flags():
    d = ProteinMPNNDesigner(repo_dir="/does/not/exist", ca_only=True)
    cmd = d._build_command("bb.pdb", "/tmp/out", n_seqs=4, temperature=0.1, seed=7, batch_size=1)
    assert "--ca_only" in cmd
    assert "--num_seq_per_target" in cmd and "4" in cmd
    assert "--pdb_path" in cmd and "bb.pdb" in cmd


def test_designer_raises_helpful_error_when_repo_missing(tmp_path):
    d = ProteinMPNNDesigner(repo_dir=tmp_path)  # no protein_mpnn_run.py inside
    try:
        d.design(tmp_path / "bb.pdb")
    except FileNotFoundError as exc:
        assert "download_proteinmpnn" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected FileNotFoundError")


# --- fallback designer + PDB helpers -----------------------------------------


def _write_ca_pdb(path: Path, length: int) -> None:
    lines = []
    for i in range(1, length + 1):
        lines.append(
            f"ATOM  {i:>5d}  CA  ALA A{i:>4d}    "
            f"{i * 1.5:>8.3f}{0.0:>8.3f}{0.0:>8.3f}  1.00  0.00           C"
        )
    lines.append("TER\nEND")
    path.write_text("\n".join(lines))


def test_pdb_ca_length(tmp_path):
    p = tmp_path / "bb.pdb"
    _write_ca_pdb(p, 42)
    assert pdb_ca_length(p) == 42


def test_fallback_designer_from_length_produces_valid_candidates():
    designer = FallbackDesigner(seed=1)
    cands = designer.design(length=40, n_seqs=5)
    assert len(cands) == 5
    for c in cands:
        assert c.length == 40
        assert c.is_valid()  # only canonical residues
        assert all(ch in AA_SET for ch in c.sequence)
        assert c.origin == "fallback"
        assert motif_satisfied(c.sequence)  # motif grafted
        assert c.metadata["motif_satisfied"] is True


def test_fallback_designer_from_pdb(tmp_path):
    p = tmp_path / "design_007.pdb"
    _write_ca_pdb(p, 30)
    cands = FallbackDesigner(seed=2).design(pdb_path=p, n_seqs=3)
    assert len(cands) == 3
    assert all(c.length == 30 for c in cands)
    assert all(c.metadata["backbone_id"] == "design_007" for c in cands)


def test_candidates_feed_oracle_types():
    # Designed candidates must be scoreable by the Phase 1 oracle types.
    from pdz_denovo.oracle import SolubilityOracle

    cands = FallbackDesigner(seed=3).design(length=50, n_seqs=2)
    sol = SolubilityOracle()
    for c in cands:
        v = sol.score(c)
        assert 0.0 <= v <= 1.0
