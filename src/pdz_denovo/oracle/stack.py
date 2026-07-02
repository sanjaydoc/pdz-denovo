"""OracleStack: combine stability, solubility, and binding into OracleScore.

Provides batched scoring (ESM stability is batched for efficiency) and optional
on-disk JSON caching keyed by sequence, so repeated candidates across DBTL
cycles are not re-scored.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from pdz_denovo.oracle.binding import BindingOracle
from pdz_denovo.oracle.solubility import SolubilityOracle
from pdz_denovo.oracle.stability import StabilityOracle
from pdz_denovo.oracle.types import Candidate, OracleScore

LOGGER = logging.getLogger("pdz_denovo")


class OracleStack:
    def __init__(
        self,
        stability: StabilityOracle,
        solubility: SolubilityOracle,
        binding: BindingOracle,
        cache: bool = True,
        cache_dir: str | Path = "outputs/oracle_cache",
    ) -> None:
        self.stability = stability
        self.solubility = solubility
        self.binding = binding
        self.cache_enabled = cache
        self.cache_dir = Path(cache_dir)
        self._cache: dict[str, dict] = {}
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_file = self.cache_dir / "scores.json"
            self._load_cache()

    # --- caching -------------------------------------------------------------
    def _load_cache(self) -> None:
        if self._cache_file.exists():
            try:
                self._cache = json.loads(self._cache_file.read_text())
                LOGGER.info("Loaded oracle cache (%d entries).", len(self._cache))
            except json.JSONDecodeError:
                LOGGER.warning("Corrupt oracle cache; starting fresh.")
                self._cache = {}

    def _save_cache(self) -> None:
        if self.cache_enabled:
            self._cache_file.write_text(json.dumps(self._cache, indent=2))

    # --- scoring -------------------------------------------------------------
    def score(self, candidate: Candidate) -> OracleScore:
        return self.score_batch([candidate])[0]

    def score_batch(self, candidates: list[Candidate]) -> list[OracleScore]:
        results: list[OracleScore | None] = [None] * len(candidates)
        to_compute: list[int] = []

        # 1. Serve from cache where possible.
        for i, c in enumerate(candidates):
            if self.cache_enabled and c.sequence in self._cache:
                d = self._cache[c.sequence]
                results[i] = OracleScore(
                    stability=d["stability"],
                    solubility=d["solubility"],
                    binding=d["binding"],
                    details=d.get("details", {}),
                )
            else:
                to_compute.append(i)

        if to_compute:
            compute_cands = [candidates[i] for i in to_compute]

            # Stability is batched (single ESM forward pass over the batch).
            stab = self.stability.score_batch(compute_cands)
            for local_idx, global_idx in enumerate(to_compute):
                c = candidates[global_idx]
                sol = self.solubility.score(c)
                bind = self.binding.score(c)
                score = OracleScore(
                    stability=float(stab[local_idx]),
                    solubility=float(sol),
                    binding=float(bind),
                    details={
                        "stability_detail": c.metadata.get("stability_detail", {}),
                        "solubility_detail": c.metadata.get("solubility_detail", {}),
                        "binding_detail": c.metadata.get("binding_detail", {}),
                    },
                )
                results[global_idx] = score
                if self.cache_enabled:
                    self._cache[c.sequence] = score.to_dict()
            self._save_cache()

        return [r for r in results if r is not None]


def build_oracle_stack(cfg) -> OracleStack:
    """Construct an OracleStack from a Hydra/OmegaConf oracle config node."""
    stability = StabilityOracle(
        esm_model=cfg.stability.esm_model,
        mode="wt_marginal",
        normalize=bool(cfg.stability.get("normalize", True)),
    )
    solubility = SolubilityOracle(
        hydrophobicity_weight=cfg.solubility.hydrophobicity_weight,
        charge_weight=cfg.solubility.charge_weight,
        aggregation_weight=cfg.solubility.aggregation_weight,
    )
    binding = BindingOracle(
        method=cfg.binding.method,
        smina_exe=cfg.binding.get("smina_exe", None),
    )
    return OracleStack(
        stability=stability,
        solubility=solubility,
        binding=binding,
        cache=bool(cfg.get("cache", True)),
        cache_dir=cfg.get("cache_dir", "outputs/oracle_cache"),
    )
