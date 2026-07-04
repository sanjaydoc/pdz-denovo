"""Phase 6 tests: random baseline, training-data id loading, and report presence.

Torch-free — covers the benchmark's baseline optimizer and the data-pipeline
helpers that don't require network access.
"""
from __future__ import annotations

from pathlib import Path

from pdz_denovo.optimize import RandomOptimizer
from pdz_denovo.sequence import FallbackDesigner, motif_satisfied

ROOT = Path(__file__).resolve().parents[1]


def test_random_optimizer_matches_length_and_motif():
    pop = FallbackDesigner(seed=0).design(length=48, n_seqs=6)
    scores = [[0.1, 0.2, 0.3]] * 6
    opt = RandomOptimizer(seed=1)
    proposals = opt.propose(pop, scores, n=10)
    assert len(proposals) == 10
    for c in proposals:
        assert c.length == 48
        assert c.origin == "random"
        assert c.is_valid()
        assert motif_satisfied(c.sequence)  # motif grafted


def test_random_optimizer_ignores_scores():
    # Baseline must not depend on the objective values (pure random search).
    pop = FallbackDesigner(seed=2).design(length=30, n_seqs=4)
    a = RandomOptimizer(seed=9).propose(pop, [[0, 0, 0]] * 4, 4)
    b = RandomOptimizer(seed=9).propose(pop, [[9, 9, 9]] * 4, 4)
    assert [c.sequence for c in a] == [c.sequence for c in b]


def test_training_data_default_ids_are_reasonable():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "download_training_data", ROOT / "scripts" / "download_training_data.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ids = mod.DEFAULT_PDB_IDS
    assert len(ids) >= 20
    assert all(len(pid) == 4 for pid in ids)  # valid PDB id format
    assert len(set(ids)) == len(ids)  # no duplicates


def test_report_and_plan_exist():
    assert (ROOT / "docs" / "REPORT.md").exists()
    assert (ROOT / "docs" / "PLAN.md").exists()
    text = (ROOT / "docs" / "REPORT.md").read_text()
    assert "hypervolume" in text.lower()
    assert "self-consistency" in text.lower()
