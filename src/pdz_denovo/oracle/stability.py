"""Stability / foldability oracle via ESM-2 pseudo-log-likelihood.

We use a protein language model (ESM-2) to estimate how "native-like" a
sequence is. Sequences that the model finds probable (high average
log-likelihood of their own residues) tend to be better-folded and more stable
-- a standard, well-supported zero-shot proxy for stability / fitness.

Two scoring modes:
  * "wt_marginal" (default, fast): a single forward pass; average log-prob of
    the true residue at each position. Cheap enough for closed-loop use on
    6 GB VRAM.
  * "masked": mask each position in turn (L forward passes); more accurate but
    L x slower. Use only for small final re-ranking.

Higher score = more stable (scores are negative log-likelihoods negated, i.e.
average log-likelihood, which is <= 0; optionally min-max normalized per batch).
"""
from __future__ import annotations

import logging

from pdz_denovo.oracle.base import BaseOracle
from pdz_denovo.oracle.types import Candidate

LOGGER = logging.getLogger("pdz_denovo")

_ESM_LOADERS = {
    "esm2_t6_8M_UR50D": "esm2_t6_8M_UR50D",
    "esm2_t12_35M_UR50D": "esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D": "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D": "esm2_t33_650M_UR50D",
}


class StabilityOracle(BaseOracle):
    name = "stability"

    def __init__(
        self,
        esm_model: str = "esm2_t12_35M_UR50D",
        device: str = "auto",
        mode: str = "wt_marginal",
        normalize: bool = True,
        max_batch_tokens: int = 4096,
    ) -> None:
        if esm_model not in _ESM_LOADERS:
            raise ValueError(
                f"Unknown ESM model '{esm_model}'. Options: {list(_ESM_LOADERS)}"
            )
        self.esm_model_name = esm_model
        self.mode = mode
        self.normalize = normalize
        self.max_batch_tokens = max_batch_tokens
        self._device = device
        self._model = None
        self._alphabet = None
        self._batch_converter = None

    # --- lazy model loading --------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import esm
        import torch

        from pdz_denovo.utils.common import resolve_device

        self._device = resolve_device(self._device)
        LOGGER.info("Loading ESM-2 model '%s' on %s ...", self.esm_model_name, self._device)
        loader = getattr(esm.pretrained, _ESM_LOADERS[self.esm_model_name])
        model, alphabet = loader()
        model.eval().to(self._device)
        # fp16 on GPU to save VRAM.
        if self._device == "cuda":
            model = model.half()
        self._model = model
        self._alphabet = alphabet
        self._batch_converter = alphabet.get_batch_converter()
        LOGGER.info("ESM-2 loaded (%s).", self.esm_model_name)

    # --- core scoring --------------------------------------------------------
    def _avg_loglik(self, sequences: list[str]) -> list[float]:
        """Average per-residue log-likelihood for each sequence (wt-marginal)."""
        import torch

        self._ensure_loaded()
        data = [(f"seq{i}", s) for i, s in enumerate(sequences)]
        _, _, tokens = self._batch_converter(data)
        tokens = tokens.to(self._device)

        with torch.no_grad():
            out = self._model(tokens, repr_layers=[], return_contacts=False)
            logits = out["logits"].float()  # (B, L+2, V)
            logp = torch.log_softmax(logits, dim=-1)

        results = []
        pad_idx = self._alphabet.padding_idx
        bos_idx = self._alphabet.cls_idx
        eos_idx = self._alphabet.eos_idx
        special = {pad_idx, bos_idx, eos_idx}
        for b, seq in enumerate(sequences):
            tok = tokens[b]
            token_logp = logp[b]
            total = 0.0
            count = 0
            for pos in range(tok.shape[0]):
                t = int(tok[pos].item())
                if t in special:
                    continue
                total += float(token_logp[pos, t].item())
                count += 1
            results.append(total / max(count, 1))
        return results

    def _masked_loglik(self, sequence: str) -> float:
        """Masked-marginal average log-likelihood (accurate, L forward passes)."""
        import torch

        self._ensure_loaded()
        mask_idx = self._alphabet.mask_idx
        data = [("seq", sequence)]
        _, _, base_tokens = self._batch_converter(data)
        base_tokens = base_tokens.to(self._device)
        # positions 1..L (excluding BOS at 0 and EOS at end)
        positions = list(range(1, base_tokens.shape[1] - 1))

        total, count = 0.0, 0
        with torch.no_grad():
            for pos in positions:
                masked = base_tokens.clone()
                true_tok = int(masked[0, pos].item())
                masked[0, pos] = mask_idx
                out = self._model(masked, repr_layers=[], return_contacts=False)
                logp = torch.log_softmax(out["logits"][0, pos].float(), dim=-1)
                total += float(logp[true_tok].item())
                count += 1
        return total / max(count, 1)

    # --- public API ----------------------------------------------------------
    def score(self, candidate: Candidate) -> float:
        return self.score_batch([candidate])[0]

    def score_batch(self, candidates: list[Candidate]) -> list[float]:
        seqs = [c.sequence for c in candidates]
        if self.mode == "masked":
            raw = [self._masked_loglik(s) for s in seqs]
        else:
            raw = self._avg_loglik(seqs)

        for c, r in zip(candidates, raw):
            c.metadata.setdefault("stability_detail", {}).update(
                {"avg_loglik": round(r, 4), "mode": self.mode}
            )

        if not self.normalize or len(raw) == 1:
            # Return raw average log-likelihood (<=0); higher = better.
            return [float(r) for r in raw]

        lo, hi = min(raw), max(raw)
        if hi - lo < 1e-8:
            return [0.5 for _ in raw]
        return [float((r - lo) / (hi - lo)) for r in raw]
