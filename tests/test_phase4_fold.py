"""Phase 4 tests for ESMFold self-consistency validation.

Dependency-light: the RMSD math and PDB parsing are numpy-only, and folding is
exercised through a fake backend so no network / ESMFold weights are needed. The
real ESMFold call is validated on the user's machine.
"""
from __future__ import annotations

import numpy as np
import pytest

from pdz_denovo.fold import (
    FoldResult,
    evaluate_design,
    kabsch_rmsd_np,
    parse_fold_pdb,
    summarize,
    validate_candidates,
)
from pdz_denovo.oracle.types import Candidate


# --- RMSD ---------------------------------------------------------------------


def test_kabsch_rmsd_zero_for_rotated_translated_copy():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(20, 3))
    # Random rotation via QR.
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    Q = P @ q.T + np.array([5.0, -2.0, 1.0])  # rotate + translate
    assert kabsch_rmsd_np(P, Q) < 1e-6


def test_kabsch_rmsd_positive_for_noisy():
    rng = np.random.default_rng(1)
    P = rng.normal(size=(30, 3))
    Q = P + rng.normal(scale=0.5, size=(30, 3))
    assert kabsch_rmsd_np(P, Q) > 0.1


# --- PDB parsing --------------------------------------------------------------


def _fake_pred_pdb(coords, plddt):
    lines = []
    for i, ((x, y, z), b) in enumerate(zip(coords, plddt), start=1):
        lines.append(
            f"ATOM  {i:>5d}  CA  ALA A{i:>4d}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00{b:>6.2f}           C"
        )
    return "\n".join(lines) + "\nEND\n"


def test_parse_fold_pdb_extracts_coords_and_plddt():
    coords = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
    plddt = [80.0, 90.0, 70.0]
    ca, b = parse_fold_pdb(_fake_pred_pdb(coords, plddt))
    assert ca.shape == (3, 3)
    assert np.allclose(ca[1], [4.0, 5.0, 6.0])
    assert np.allclose(b, [80.0, 90.0, 70.0])


def test_parse_fold_pdb_raises_without_ca():
    with pytest.raises(ValueError):
        parse_fold_pdb("HEADER only\nEND\n")


def test_parse_fold_pdb_rescales_0_to_1_plddt():
    # esmatlas returns pLDDT on a 0-1 scale; it must be normalised to 0-100.
    coords = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
    _, b = parse_fold_pdb(_fake_pred_pdb(coords, [0.32, 0.93]))
    assert np.allclose(b, [32.0, 93.0])


# --- fake fold backend --------------------------------------------------------


class _FakeBackend:
    """Returns the designed backbone (optionally perturbed) as the 'fold'."""

    def __init__(self, coords, noise=0.0, plddt=85.0, seed=0):
        self.coords = np.asarray(coords, dtype="float32")
        self.noise = noise
        self.plddt = plddt
        self.rng = np.random.default_rng(seed)

    def fold(self, sequence):
        pred = self.coords + self.noise * self.rng.normal(size=self.coords.shape)
        L = len(pred)
        return FoldResult(
            sequence=sequence,
            ca_coords=pred.astype("float32"),
            plddt_per_res=np.full(L, self.plddt, dtype="float32"),
            mean_plddt=self.plddt,
        )


def test_evaluate_design_perfect_is_success():
    coords = np.random.default_rng(2).normal(size=(40, 3))
    backend = _FakeBackend(coords, noise=0.0, plddt=90.0)
    m = evaluate_design(coords, "A" * 40, backend)
    assert m["scRMSD"] < 1e-4
    assert m["success"] is True


def test_evaluate_design_low_plddt_fails():
    coords = np.random.default_rng(3).normal(size=(40, 3))
    backend = _FakeBackend(coords, noise=0.0, plddt=40.0)  # confident-looking but low pLDDT
    m = evaluate_design(coords, "A" * 40, backend)
    assert m["scRMSD"] < 1e-4
    assert m["success"] is False  # gated by pLDDT


def test_validate_candidates_annotates_and_sorts():
    coords = np.random.default_rng(4).normal(size=(30, 3))
    backend = _FakeBackend(coords, noise=0.3, plddt=80.0, seed=5)
    cands = [Candidate(sequence="ACDEFGHIKLMNPQRSTVWYACDEFGHIKL") for _ in range(3)]
    results = validate_candidates(coords, cands, backend, backbone_id="design_000")
    assert len(results) == 3
    # Sorted ascending by scRMSD.
    scr = [r["scRMSD"] for r in results]
    assert scr == sorted(scr)
    # Each candidate got its metadata annotated.
    for c in cands:
        assert "self_consistency" in c.metadata
        assert c.metadata["self_consistency"]["plddt"] is not None


def test_summarize_counts_success():
    results = [
        {"scRMSD": 1.0, "plddt": 85.0, "success": True},
        {"scRMSD": 3.0, "plddt": 60.0, "success": False},
        {"scRMSD": None, "plddt": None, "success": False, "error": "x"},
    ]
    s = summarize(results)
    assert s["n"] == 3 and s["n_scored"] == 2 and s["n_success"] == 1
    assert s["best_scRMSD"] == 1.0
