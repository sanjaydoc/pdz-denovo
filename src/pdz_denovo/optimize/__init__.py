"""Multi-objective optimization — propose the next DBTL library (Phase 5).

NSGA-II (genetic) and qNEHVI (Bayesian) optimizers plus the Pareto/hypervolume
math. The Pareto module and NSGA-II are torch-free; qNEHVI lazily requires
BoTorch.
"""
from pdz_denovo.optimize.nsga2 import NSGA2Optimizer
from pdz_denovo.optimize.pareto import (
    crowding_distance,
    dominates,
    hypervolume,
    non_dominated_sort,
    pareto_front_indices,
)

__all__ = [
    "NSGA2Optimizer",
    "dominates",
    "non_dominated_sort",
    "pareto_front_indices",
    "crowding_distance",
    "hypervolume",
]
