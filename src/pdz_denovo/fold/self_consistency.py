"""Self-consistency validation — the credibility gate of the pipeline.

The metric the de novo design literature (RFdiffusion, FrameDiff, FrameFlow)
uses to decide whether a design is "real":

    designed backbone  --ProteinMPNN-->  sequence  --ESMFold-->  refolded structure

If the refolded structure matches the designed backbone, the design is
**self-consistent**. We quantify the match with:

* **scRMSD** — Cα RMSD between the ESMFold prediction and the designed backbone
  after optimal superposition (Kabsch). Success threshold ``< 2 Å``.
* **pLDDT** — ESMFold's mean confidence. Success threshold ``> 70``.

Because folding is expensive, this runs on the **top-K** designs per cycle, not
the whole library — which is exactly the "spend the assay budget on promising
candidates" behaviour a sparse/high-cost-data campaign requires.

The RMSD math is numpy-only (no torch), so validation runs even on an API-only
setup with no local model.
"""
from __future__ import annotations

import logging

LOGGER = logging.getLogger("pdz_denovo")

SCRMSD_SUCCESS = 2.0   # Å
PLDDT_SUCCESS = 70.0   # ESMFold confidence


def kabsch_rmsd_np(P, Q) -> float:
    """Minimum Cα RMSD between two ``(L, 3)`` point sets (numpy, Kabsch)."""
    import numpy as np

    P = np.asarray(P, dtype="float64")
    Q = np.asarray(Q, dtype="float64")
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    h = Pc.T @ Qc
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    D = np.diag([1.0, 1.0, d])
    r = vt.T @ D @ u.T
    P_aligned = Pc @ r.T
    return float(np.sqrt(np.mean(np.sum((P_aligned - Qc) ** 2, axis=1))))


def evaluate_design(designed_ca, sequence: str, fold_backend) -> dict:
    """Fold one sequence and score it against the designed backbone.

    Args:
        designed_ca: ``(L, 3)`` Cα coordinates of the designed backbone.
        sequence: the ProteinMPNN sequence for that backbone.
        fold_backend: object with ``.fold(seq) -> FoldResult``.

    Returns:
        A dict with ``scRMSD``, ``plddt``, ``success`` and lengths.
    """
    import numpy as np

    result = fold_backend.fold(sequence)
    folded_ca = np.asarray(result.ca_coords)
    designed_ca = np.asarray(designed_ca)
    n = min(len(folded_ca), len(designed_ca))
    scrmsd = kabsch_rmsd_np(folded_ca[:n], designed_ca[:n])
    plddt = float(result.mean_plddt)
    return {
        "scRMSD": round(scrmsd, 3),
        "plddt": round(plddt, 2),
        "length": int(n),
        "success": bool(scrmsd < SCRMSD_SUCCESS and plddt > PLDDT_SUCCESS),
    }


def validate_candidates(designed_ca, candidates, fold_backend, backbone_id: str = "") -> list:
    """Fold each candidate for a backbone and annotate self-consistency.

    Mutates each candidate's ``metadata['self_consistency']`` and returns the
    list of per-candidate result dicts, sorted best (lowest scRMSD) first.
    """
    results = []
    for c in candidates:
        try:
            metrics = evaluate_design(designed_ca, c.sequence, fold_backend)
        except Exception as exc:  # noqa: BLE001 - one bad fold shouldn't abort
            LOGGER.warning("Fold failed for %s: %s", c.id, exc)
            metrics = {"scRMSD": None, "plddt": None, "success": False, "error": str(exc)}
        c.metadata["self_consistency"] = metrics
        results.append({"id": c.id, "backbone_id": backbone_id, **metrics})
    results.sort(key=lambda r: (r["scRMSD"] is None, r["scRMSD"] if r["scRMSD"] is not None else 1e9))
    return results


def summarize(results: list) -> dict:
    """Aggregate a batch of validation results into headline numbers."""
    scored = [r for r in results if r.get("scRMSD") is not None]
    n_success = sum(1 for r in scored if r["success"])
    best = scored[0] if scored else None
    return {
        "n": len(results),
        "n_scored": len(scored),
        "n_success": n_success,
        "success_rate": round(n_success / len(results), 3) if results else 0.0,
        "best_scRMSD": best["scRMSD"] if best else None,
        "best_plddt": best["plddt"] if best else None,
    }
