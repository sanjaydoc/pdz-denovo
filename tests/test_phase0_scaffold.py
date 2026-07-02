"""Phase 0 smoke tests: verify scaffold, configs, and utilities load correctly.

These tests intentionally avoid heavy dependencies (torch/esm) so they can run
immediately after `pip install -r requirements.txt` without a GPU.
"""
from __future__ import annotations

from pathlib import Path

from pdz_denovo.utils.common import (
    get_project_root,
    resolve_device,
    set_seed,
    setup_logging,
)

ROOT = get_project_root()


def test_project_root_exists():
    assert ROOT.exists()
    assert (ROOT / "configs" / "config.yaml").exists()


def test_config_layout_present():
    for rel in [
        "configs/config.yaml",
        "configs/model/flow.yaml",
        "configs/oracle/default.yaml",
        "configs/optimize/moo.yaml",
    ]:
        assert (ROOT / rel).exists(), f"missing config: {rel}"


def test_master_config_loads_with_omegaconf():
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(ROOT / "configs" / "config.yaml")
    assert cfg.target.pdb_id == "1BE9"
    assert cfg.design.min_length <= cfg.design.max_length


def test_setup_logging_idempotent():
    log1 = setup_logging("INFO")
    log2 = setup_logging("DEBUG")
    assert log1 is log2  # same singleton logger
    assert len(log1.handlers) == 1  # no duplicate handlers


def test_set_seed_runs():
    set_seed(123)  # should not raise even without torch


def test_resolve_device_returns_valid():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in ("cpu", "cuda")
