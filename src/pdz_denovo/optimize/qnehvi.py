"""BoTorch qNEHVI multi-objective Bayesian optimization for library selection.

The "advanced" optimizer path the JD names explicitly. Instead of evolving
sequences (NSGA-II), qNEHVI fits a Gaussian-process surrogate to the evaluated
(features -> objectives) data and selects the next library by maximizing the
**q-Noisy Expected Hypervolume Improvement** acquisition over a discrete pool of
candidate sequences.

This is *batch* Bayesian optimization for multi-objective, high-cost data —
exactly the "closed-loop optimization backbone over sparse, noisy, expensive
assays" the role is about. Features come from :mod:`.features` (ESM-2
embeddings); the candidate pool is typically NSGA-II-style mutants of the current
front.

Requires BoTorch/GPyTorch (in requirements.txt). Imports are guarded so the rest
of the package works without them.
"""
from __future__ import annotations

import logging

LOGGER = logging.getLogger("pdz_denovo")


class QNEHVISelector:
    """Select the next library via qNEHVI over a discrete candidate pool."""

    def __init__(
        self,
        ref_point,
        mc_samples: int = 64,
        seed: int = 0,
    ) -> None:
        self.ref_point = list(ref_point)
        self.mc_samples = mc_samples
        self.seed = seed

    def select(self, train_X, train_Y, pool_X, n: int) -> list[int]:
        """Return indices into ``pool_X`` of the ``n`` selected candidates.

        Args:
            train_X: ``(M, D)`` features of already-evaluated candidates.
            train_Y: ``(M, K)`` objective values (maximization).
            pool_X: ``(P, D)`` features of candidate proposals to choose from.
            n: number to select.
        """
        try:
            import torch
            from botorch.acquisition.multi_objective.monte_carlo import (
                qNoisyExpectedHypervolumeImprovement,
            )
            from botorch.models import SingleTaskGP
            from botorch.models.model_list_gp_regression import ModelListGP
            from botorch.models.transforms.input import Normalize
            from botorch.models.transforms.outcome import Standardize
            from botorch.optim import optimize_acqf_discrete
            from botorch.sampling.normal import SobolQMCNormalSampler
            from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
            from botorch.fit import fit_gpytorch_mll
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                "qNEHVI requires botorch/gpytorch. Install requirements.txt, or "
                "use the NSGA-II optimizer instead."
            ) from exc

        torch.manual_seed(self.seed)
        X = torch.as_tensor(train_X, dtype=torch.double)
        Y = torch.as_tensor(train_Y, dtype=torch.double)
        pool = torch.as_tensor(pool_X, dtype=torch.double)

        # One independent GP per objective (all maximization). Normalize the
        # (high-dim ESM) features to the unit cube and standardize outcomes, so
        # the GP length scales are well conditioned — without this the surrogate
        # sees raw embeddings and warns "input not in unit cube".
        d = X.shape[-1]
        models = [
            SingleTaskGP(
                X,
                Y[:, k : k + 1],
                input_transform=Normalize(d=d),
                outcome_transform=Standardize(m=1),
            )
            for k in range(Y.shape[-1])
        ]
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.mc_samples]))
        acqf = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=torch.as_tensor(self.ref_point, dtype=torch.double),
            X_baseline=X,
            sampler=sampler,
            prune_baseline=True,
        )
        # Sequential-greedy selection over the discrete pool.
        candidates, _ = optimize_acqf_discrete(
            acq_function=acqf, q=min(n, pool.shape[0]), choices=pool, unique=True
        )
        # Map chosen rows back to pool indices.
        chosen = []
        for c in candidates:
            diffs = (pool - c).abs().sum(dim=-1)
            chosen.append(int(diffs.argmin().item()))
        return chosen


class QNEHVIProposer:
    """DBTL-loop adapter: qNEHVI selection over a mutation-generated pool.

    Implements the ``propose(population, scores, n)`` interface the loop expects.
    It accumulates all evaluated (features, objectives) across cycles as the GP
    training set, builds a candidate pool by mutating the current population
    (reusing the NSGA-II operators), embeds it, and selects the next library with
    qNEHVI.
    """

    def __init__(
        self,
        featurizer,
        ref_point,
        mutation_rate: float = 0.05,
        crossover_rate: float = 0.7,
        pool_multiplier: int = 6,
        seed: int = 0,
    ) -> None:
        from pdz_denovo.optimize.nsga2 import NSGA2Optimizer

        self.featurizer = featurizer
        self.selector = QNEHVISelector(ref_point, seed=seed)
        self._pool_gen = NSGA2Optimizer(
            mutation_rate=mutation_rate, crossover_rate=crossover_rate, seed=seed
        )
        self.pool_multiplier = pool_multiplier
        self._X: list = []
        self._Y: list = []

    def propose(self, population, scores, n: int) -> list:
        import numpy as np

        feats = self.featurizer.embed([c.sequence for c in population])
        self._X.extend(list(feats))
        self._Y.extend([list(s) for s in scores])

        pool = self._pool_gen.propose(population, scores, n * self.pool_multiplier)
        pool_feats = self.featurizer.embed([c.sequence for c in pool])
        idx = self.selector.select(
            np.asarray(self._X), np.asarray(self._Y), np.asarray(pool_feats), n
        )
        return [pool[i] for i in idx]
