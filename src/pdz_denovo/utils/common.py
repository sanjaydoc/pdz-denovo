"""Utility helpers: device selection, seeding, logging, and paths."""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np

_LOGGER_CONFIGURED = False


def get_project_root() -> Path:
    """Return the repository root (two levels above this package)."""
    return Path(__file__).resolve().parents[3]


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure a single rich-friendly root logger (idempotent)."""
    global _LOGGER_CONFIGURED
    logger = logging.getLogger("pdz_denovo")
    if not _LOGGER_CONFIGURED:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
        handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
        _LOGGER_CONFIGURED = True
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and (if available) PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(device: str = "auto") -> str:
    """Resolve the compute device string.

    Args:
        device: "auto", "cuda", or "cpu".

    Returns:
        "cuda" if requested/available, otherwise "cpu".
    """
    if device == "cpu":
        return "cpu"
    try:
        import torch

        if device in ("auto", "cuda") and torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"
