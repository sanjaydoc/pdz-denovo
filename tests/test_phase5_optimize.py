"""Phase 5 tests: Pareto math, NSGA-II, the DBTL loop, and tracking.

All torch-free: the loop is exercised with a deterministic mock scorer, so the
closed-loop logic (archive growth, hypervolume monotonicity, next-library
proposal) is fully covered without ESM-2/BoTorch.
"""
from __future__ import annotations

import json

from pdz_denovo.loop import DBTLoop
from pdz_denovo.optimize import (
    NSGA2Optimizer,
    dominates,
    hypervolume,
    non_dominated_sort,
    pareto_front_indices,
)
from pdz_denovo.optimize.pareto import crowding_distance
from pdz_denovo.sequence import FallbackDesigner, motif_satisfied
from pdz_denovo.tracking import RunTracker


# --- Pareto ------------------------------------------------------------------


def test_dominates_maximization():
    assert dominates([1.0, 1.0], [0.5, 0.5])
    assert dominates([1.0, 0.5], [0.5, 0.5])
    assert not dominates([1.0, 0.0], [0.0, 1.0])  # trade-off, non-dominated
    assert not dominates([0.5, 0.5], [0.5, 0.5])  # equal, not strict


def test_non_dominated_sort_fronts():
    scores = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.4, 0.4], [0.1, 0.1]]
    fronts = non_dominated_sort(scores)
    assert set(fronts[0]) == {0, 1, 2}  # mutually non-dominated
    assert fronts[1] == [3]
    assert fronts[2] == [4]


def test_pareto_front_indices():
    scores = [[2, 1], [1, 2], [0, 0], [1.5, 1.5]]
    pf = set(pareto_front_indices(scores))
    assert pf == {0, 1, 3}


def test_crowding_distance_boundaries_infinite():
    scores = [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]
    cd = crowding_distance(scores, [0, 1, 2])
    assert cd[0] == float("inf") and cd[2] == float("inf")
    assert cd[1] < float("inf")


def test_hypervolume_2d_known_value():
    # Two points (maximize), ref at origin.
    # Points (2,1) and (1,2), ref (0,0): union area = 2*1 + 1*2 - 1*1 = 3.
    hv = hypervolume([[2.0, 1.0], [1.0, 2.0]], ref=[0.0, 0.0], maximize=True)
    assert abs(hv - 3.0) < 1e-9


def test_hypervolume_3d_single_point():
    hv = hypervolume([[2.0, 3.0, 4.0]], ref=[0.0, 0.0, 0.0], maximize=True)
    assert abs(hv - 24.0) < 1e-9


def test_hypervolume_monotonic_when_adding_better_point():
    ref = [0.0, 0.0]
    base = [[1.0, 1.0]]
    more = [[1.0, 1.0], [2.0, 2.0]]
    assert hypervolume(more, ref) >= hypervolume(base, ref)


def test_hypervolume_3d_dominated_point_not_double_counted():
    # (1,1,1) is dominated by (2,2,2) (maximization) -> HV must be unchanged.
    ref = [0.0, 0.0, 0.0]
    a = hypervolume([[2.0, 2.0, 2.0]], ref)
    b = hypervolume([[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]], ref)
    assert abs(a - b) < 1e-9 and abs(a - 8.0) < 1e-9


def test_hypervolume_3d_monotonic_under_incremental_addition():
    # Regression: a slice with a 2D-dominated projection must not shrink HV.
    ref = [0.0, 0.0, 0.0]
    pts = [[5, 5, 1], [1, 1, 5], [2, 2, 2], [4, 1, 1], [1, 4, 3], [3, 3, 3]]
    prev, cur = 0.0, []
    for p in pts:
        cur.append([float(v) for v in p])
        hv = hypervolume(cur, ref)
        assert hv >= prev - 1e-9, f"HV decreased at {p}: {hv} < {prev}"
        prev = hv


# --- NSGA-II -----------------------------------------------------------------


def _cands(seqs):
    from pdz_denovo.oracle.types import Candidate

    return [Candidate(sequence=s) for s in seqs]


def test_nsga2_propose_count_and_validity():
    pop = FallbackDesigner(seed=0).design(length=30, n_seqs=8)
    scores = [[i / 8, 1 - i / 8, 0.5] for i in range(8)]
    opt = NSGA2Optimizer(seed=1)
    offspring = opt.propose(pop, scores, n=12)
    assert len(offspring) == 12
    for c in offspring:
        assert c.is_valid()
        assert c.origin == "nsga2"
        assert motif_satisfied(c.sequence)  # motif preserved through mutation


def test_nsga2_is_deterministic_with_seed():
    pop = FallbackDesigner(seed=0).design(length=25, n_seqs=6)
    scores = [[i, 6 - i, 1.0] for i in range(6)]
    a = NSGA2Optimizer(seed=7).propose(pop, scores, 6)
    b = NSGA2Optimizer(seed=7).propose(pop, scores, 6)
    assert [c.sequence for c in a] == [c.sequence for c in b]


# --- DBTL loop ---------------------------------------------------------------


def _mock_scorer(cands):
    """Deterministic 3-objective score from sequence composition."""
    out = []
    for c in cands:
        s = c.sequence
        L = max(len(s), 1)
        out.append([
            s.count("E") / L,
            s.count("K") / L,
            1.0 - s.count("A") / L,
        ])
    return out


def test_dbtl_loop_runs_and_hypervolume_is_monotonic():
    designer = FallbackDesigner(seed=3)
    loop = DBTLoop(
        scorer=_mock_scorer,
        optimizer=NSGA2Optimizer(seed=3),
        seed_fn=lambda n: designer.design(length=30, n_seqs=n),
        ref_point=[0.0, 0.0, 0.0],
    )
    result = loop.run(n_cycles=4, library_size=8, n_seed=8)
    history = result["history"]
    assert len(history) == 4
    # Archive grows every cycle.
    sizes = [h["archive_size"] for h in history]
    assert sizes == sorted(sizes) and sizes[0] == 8
    # Cumulative archive + fixed ref => hypervolume never decreases.
    hvs = [h["hypervolume"] for h in history]
    assert all(hvs[i + 1] >= hvs[i] - 1e-9 for i in range(len(hvs) - 1))
    assert len(result["pareto"]) >= 1


def test_run_tracker_writes_json(tmp_path):
    tracker = RunTracker(out_dir=tmp_path, use_mlflow=False)
    tracker.log_cycle({"cycle": 0, "hypervolume": 1.0, "best_per_objective": [0.1, 0.2, 0.3]})
    tracker.log_cycle({"cycle": 1, "hypervolume": 2.0, "best_per_objective": [0.2, 0.3, 0.4]})
    tracker.finish({"history": []})
    history = json.loads((tmp_path / "history.json").read_text())
    assert len(history) == 2 and history[1]["hypervolume"] == 2.0
    assert (tmp_path / "summary.json").exists()
