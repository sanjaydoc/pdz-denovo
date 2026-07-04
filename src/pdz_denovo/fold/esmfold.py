"""ESMFold backend for in-silico validation (Phase 4).

Structure prediction is the "expensive assay" of the loop: we fold a designed
sequence and check whether it folds *back* to the backbone we designed it for
(self-consistency). ESMFold (Lin et al., 2023) is the field-standard tool for
this in-silico check.

Two backends behind one interface, chosen for a 6 GB / 16 GB laptop:

* ``api`` (default) — POST the sequence to the public ESMFold server
  (``api.esmatlas.com``). Zero local VRAM/RAM; handles sequences up to ~400 aa.
  Ideal given the hardware budget.
* ``local`` — load ``esm.pretrained.esmfold_v1`` and run ``infer_pdb``. The full
  model is ~2.8B params (~5.6 GB fp16), so this is CPU-only on this hardware and
  slow; use only if the API is unavailable.

Both return a :class:`FoldResult` carrying the predicted Cα coordinates and the
per-residue pLDDT (parsed from the PDB B-factor column).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

LOGGER = logging.getLogger("pdz_denovo")

ESMFOLD_API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"


@dataclass
class FoldResult:
    """Predicted structure summary for one sequence."""

    sequence: str
    ca_coords: object  # np.ndarray (L, 3)
    plddt_per_res: object  # np.ndarray (L,)
    mean_plddt: float
    pdb: str = ""
    ptm: float | None = None
    extra: dict = field(default_factory=dict)


def parse_fold_pdb(pdb_text: str):
    """Extract Cα coordinates and per-residue pLDDT from a predicted PDB.

    ESMFold writes the pLDDT confidence into the B-factor column, so the CA
    B-factors are the per-residue pLDDT.

    Returns:
        ``(ca_coords, plddt_per_res)`` as numpy arrays of shape ``(L, 3)`` and
        ``(L,)``.
    """
    import numpy as np

    coords, plddt = [], []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            b = float(line[60:66])
            coords.append((x, y, z))
            plddt.append(b)
    if not coords:
        raise ValueError("No Cα atoms found in predicted PDB.")
    return np.asarray(coords, dtype="float32"), np.asarray(plddt, dtype="float32")


class ESMFoldBackend:
    """Fold sequences via the ESMFold API (default) or a local model."""

    def __init__(
        self,
        method: str = "api",
        api_url: str = ESMFOLD_API_URL,
        timeout: int = 180,
        chunk_size: int | None = 128,
        max_retries: int = 4,
        backoff: float = 4.0,
    ) -> None:
        if method not in ("api", "local"):
            raise ValueError("method must be 'api' or 'local'")
        self.method = method
        self.api_url = api_url
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.backoff = backoff
        self._model = None  # lazy local model

    # --- API backend ---------------------------------------------------------
    def _fold_api(self, sequence: str) -> str:
        import time

        import requests

        sequence = sequence.strip()
        last_error = "unknown"
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.api_url, data=sequence, timeout=self.timeout,
                    headers={"Content-Type": "text/plain"},
                )
                if resp.status_code == 200 and resp.text.startswith(
                    ("HEADER", "ATOM", "MODEL")
                ):
                    return resp.text
                last_error = f"HTTP {resp.status_code}: {resp.text[:150]}"
            except requests.exceptions.RequestException as exc:
                last_error = str(exc)
            # The public server frequently 504s / cold-starts; back off and retry.
            if attempt < self.max_retries - 1:
                wait = self.backoff * (2**attempt)
                LOGGER.warning(
                    "ESMFold API attempt %d/%d failed (%s); retrying in %.0fs ...",
                    attempt + 1, self.max_retries, last_error, wait,
                )
                time.sleep(wait)
        raise RuntimeError(
            f"ESMFold API failed after {self.max_retries} attempts: {last_error}"
        )

    # --- local backend -------------------------------------------------------
    def _ensure_local(self) -> None:
        if self._model is not None:
            return
        import esm
        import torch

        LOGGER.info("Loading local ESMFold (heavy; CPU on 6 GB hardware) ...")
        model = esm.pretrained.esmfold_v1()
        model = model.eval()
        if self.chunk_size:
            model.set_chunk_size(self.chunk_size)
        self._model = model

    def _fold_local(self, sequence: str) -> str:
        import torch

        self._ensure_local()
        with torch.no_grad():
            return self._model.infer_pdb(sequence)

    # --- public API ----------------------------------------------------------
    def fold(self, sequence: str) -> FoldResult:
        """Predict the structure of ``sequence`` and summarise it."""
        import numpy as np

        pdb = self._fold_api(sequence) if self.method == "api" else self._fold_local(sequence)
        ca, plddt = parse_fold_pdb(pdb)
        return FoldResult(
            sequence=sequence,
            ca_coords=ca,
            plddt_per_res=plddt,
            mean_plddt=float(np.mean(plddt)),
            pdb=pdb,
        )

    def fold_to_file(self, sequence: str, path: str | Path) -> FoldResult:
        """Fold and also write the predicted PDB to ``path``."""
        result = self.fold(sequence)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(result.pdb)
        return result
