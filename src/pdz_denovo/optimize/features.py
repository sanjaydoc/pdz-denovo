"""Sequence featurization for the Bayesian-optimization surrogate.

qNEHVI needs a continuous representation of each candidate to fit a GP surrogate
over. We reuse the ESM-2 model already in the stack: the mean of the final-layer
per-residue embeddings gives a fixed-size vector per sequence (transfer learning
— a pretrained PLM as the feature extractor). Loaded lazily and cached so the
DBTL loop does not re-embed repeated candidates.
"""
from __future__ import annotations

import logging

LOGGER = logging.getLogger("pdz_denovo")

# Embedding dimension / representation layer per ESM-2 model.
_ESM_REPR_LAYER = {
    "esm2_t6_8M_UR50D": 6,
    "esm2_t12_35M_UR50D": 12,
    "esm2_t30_150M_UR50D": 30,
    "esm2_t33_650M_UR50D": 33,
}


class ESMFeaturizer:
    """Mean-pooled ESM-2 embeddings as GP-surrogate features."""

    def __init__(self, esm_model: str = "esm2_t12_35M_UR50D", device: str = "auto") -> None:
        self.esm_model_name = esm_model
        self._device = device
        self._model = None
        self._alphabet = None
        self._batch_converter = None
        self._cache: dict[str, object] = {}

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import esm

        from pdz_denovo.utils.common import resolve_device

        self._device = resolve_device(self._device)
        loader = getattr(esm.pretrained, self.esm_model_name)
        model, alphabet = loader()
        model.eval().to(self._device)
        self._model = model
        self._alphabet = alphabet
        self._batch_converter = alphabet.get_batch_converter()
        self._repr_layer = _ESM_REPR_LAYER[self.esm_model_name]

    def embed(self, sequences: list[str]):
        """Return an ``(N, D)`` numpy array of mean-pooled embeddings."""
        import numpy as np
        import torch

        self._ensure_loaded()
        todo = [s for s in sequences if s not in self._cache]
        if todo:
            data = [(f"s{i}", s) for i, s in enumerate(todo)]
            _, _, tokens = self._batch_converter(data)
            tokens = tokens.to(self._device)
            with torch.no_grad():
                out = self._model(tokens, repr_layers=[self._repr_layer])
                reps = out["representations"][self._repr_layer]  # (N, L+2, D)
            for i, s in enumerate(todo):
                # Mean over real residues (exclude BOS/EOS/pad).
                L = len(s)
                vec = reps[i, 1 : L + 1].mean(dim=0).float().cpu().numpy()
                self._cache[s] = vec
        return np.stack([self._cache[s] for s in sequences])
