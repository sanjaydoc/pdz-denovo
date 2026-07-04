"""In-silico validation via ESMFold self-consistency (Phase 4).

Folds designed sequences and measures whether they refold to the designed
backbone (scRMSD) with high confidence (pLDDT) — the credibility gate used
throughout the de novo design literature. Torch-free at import (the RMSD math is
numpy; ESMFold runs behind the API by default).
"""
from pdz_denovo.fold.esmfold import ESMFoldBackend, FoldResult, parse_fold_pdb
from pdz_denovo.fold.self_consistency import (
    PLDDT_SUCCESS,
    SCRMSD_SUCCESS,
    evaluate_design,
    kabsch_rmsd_np,
    summarize,
    validate_candidates,
)

__all__ = [
    "ESMFoldBackend",
    "FoldResult",
    "parse_fold_pdb",
    "evaluate_design",
    "kabsch_rmsd_np",
    "validate_candidates",
    "summarize",
    "SCRMSD_SUCCESS",
    "PLDDT_SUCCESS",
]
