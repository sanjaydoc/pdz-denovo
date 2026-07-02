"""Phase 1 tests for the oracle stack.

The lightweight oracles (solubility, binding) and the Candidate/OracleScore
types are tested without heavy deps. The ESM-2 stability oracle is tested
behind a marker so it can be skipped on machines without the weights.
"""
from __future__ import annotations

import pytest

from pdz_denovo.oracle import (
    BindingOracle,
    Candidate,
    OracleScore,
    SolubilityOracle,
)

# --- Candidate / types -------------------------------------------------------


def test_candidate_validation_and_id():
    c = Candidate(sequence="acdefg")  # lowercase -> upper
    assert c.sequence == "ACDEFG"
    assert c.is_valid()
    assert c.id and len(c.id) == 12
    assert c.length == 6

    bad = Candidate(sequence="ACDXZ1")
    assert not bad.is_valid()


def test_oracle_score_vector_order():
    s = OracleScore(stability=0.1, solubility=0.2, binding=0.3)
    assert s.as_vector() == [0.1, 0.2, 0.3]


# --- Solubility oracle -------------------------------------------------------


def test_solubility_hydrophilic_beats_hydrophobic():
    sol = SolubilityOracle()
    # A charged, hydrophilic sequence should score higher than a poly-Ile block.
    hydrophilic = Candidate(sequence="EKEKEKDKDKEKEKDKEKEK")
    hydrophobic = Candidate(sequence="IIIIIIIIIIIIIIIIIIII")
    assert sol.score(hydrophilic) > sol.score(hydrophobic)


def test_solubility_bounded():
    sol = SolubilityOracle()
    for seq in ["ACDEFGHIKL", "MMMMMMMMMM", "EKEKEKEKEK"]:
        v = sol.score(Candidate(sequence=seq))
        assert 0.0 <= v <= 1.0


# --- Binding oracle (PDZ motif) ---------------------------------------------


def test_binding_prefers_class1_pdz_motif():
    binder = BindingOracle(method="motif")
    # Class I PSD-95 motif: ...S/T-X-V(COOH). "...TSV" is a canonical strong binder.
    strong = Candidate(sequence="GLGFKESTSV")   # ends ...S T? check: -2=T? actually 'TSV'
    weak = Candidate(sequence="GLGFKESDDD")     # acidic, non-hydrophobic C-term
    assert binder.score(strong) > binder.score(weak)


def test_binding_bounded_and_short_safe():
    binder = BindingOracle(method="motif")
    assert 0.0 <= binder.score(Candidate(sequence="ETSV")) <= 1.0
    assert binder.score(Candidate(sequence="A")) == 0.0


# --- ESM stability oracle (heavy; opt-in) -----------------------------------


@pytest.mark.esm
def test_stability_oracle_smoke():
    """Loads the small ESM-2 (8M) model and scores two sequences.

    Skipped unless run with:  pytest -m esm
    Requires downloading ~30-150 MB of weights on first use.
    """
    from pdz_denovo.oracle import StabilityOracle

    oracle = StabilityOracle(esm_model="esm2_t6_8M_UR50D", normalize=True)
    cands = [
        Candidate(sequence="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"),
        Candidate(sequence="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
    ]
    scores = oracle.score_batch(cands)
    assert len(scores) == 2
    assert all(0.0 <= s <= 1.0 for s in scores)  # normalized
