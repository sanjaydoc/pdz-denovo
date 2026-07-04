"""Backbone datasets for training the flow-matching generator.

Two datasets are provided:

* :class:`PDBBackboneDataset` — loads Cα traces from real PDB files (via
  ``biotite``), crops/pads them to a fixed length, and centres them. This is
  what you train on for the real portfolio results (download short single-domain
  chains into ``data/processed`` first).
* :class:`SyntheticBackboneDataset` — generates idealised α-helical / noisy
  backbones with no heavy dependencies. It lets the training loop and tests run
  end-to-end *before* any data is downloaded, which keeps CI green and makes the
  code inspectable on a laptop with no GPU.

Both yield fixed-length, centred ``(L, 3)`` float tensors so batches collate
trivially — a standard "random crop" strategy for backbone generative models.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

LOGGER = logging.getLogger("pdz_denovo")


class SyntheticBackboneDataset:
    """Idealised backbones for smoke-testing without downloaded data.

    Each sample is an α-helix (rise ~1.5 Å, ~100° turn per residue) with a small
    amount of Gaussian jitter and a random global rotation, so the generator has
    a non-trivial but well-defined distribution to learn.
    """

    def __init__(self, n: int = 256, length: int = 64, noise: float = 0.3, seed: int = 0) -> None:
        self.n = n
        self.length = length
        self.noise = noise
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        import torch

        from pdz_denovo.generative.frames import centre, random_rotation

        g = torch.Generator().manual_seed(self.seed + i)
        idx = torch.arange(self.length, dtype=torch.float32)
        # Canonical α-helix parameters.
        radius, rise, turn = 2.3, 1.5, math.radians(100.0)
        x = radius * torch.cos(turn * idx)
        y = radius * torch.sin(turn * idx)
        z = rise * idx
        coords = torch.stack([x, y, z], dim=-1)
        coords = coords + self.noise * torch.randn(self.length, 3, generator=g)
        coords = centre(coords)
        r = random_rotation(1, dtype=coords.dtype)[0]
        coords = coords @ r.transpose(-1, -2)
        return coords


class PDBBackboneDataset:
    """Cα traces cropped from real PDB files in a directory.

    Args:
        pdb_dir: directory containing ``*.pdb`` files (e.g. ``data/processed``).
        length: fixed crop length; chains shorter than this are skipped.
        chain: optional chain id filter.
    """

    def __init__(self, pdb_dir: str | Path, length: int = 64, chain: str | None = None) -> None:
        self.pdb_dir = Path(pdb_dir)
        self.length = length
        self.chain = chain
        self._chains = self._load_ca_traces()
        if not self._chains:
            LOGGER.warning(
                "No PDB chains >= %d residues found in %s. "
                "Falling back is the caller's responsibility.",
                length,
                self.pdb_dir,
            )

    def _load_ca_traces(self) -> list:
        import biotite.structure as struc
        import biotite.structure.io.pdb as pdb
        import numpy as np

        traces = []
        for path in sorted(self.pdb_dir.glob("*.pdb")):
            try:
                structure = pdb.PDBFile.read(str(path)).get_structure(model=1)
            except Exception as exc:  # noqa: BLE001 - skip unreadable files
                LOGGER.warning("Skipping %s (%s)", path.name, exc)
                continue
            mask = struc.filter_amino_acids(structure) & (structure.atom_name == "CA")
            if self.chain is not None:
                mask = mask & (structure.chain_id == self.chain)
            ca = structure[mask]
            if ca.array_length() >= self.length:
                traces.append(np.asarray(ca.coord, dtype="float32"))
        LOGGER.info("Loaded %d usable chains from %s.", len(traces), self.pdb_dir)
        return traces

    def __len__(self) -> int:
        return len(self._chains)

    def __getitem__(self, i: int):
        import torch

        from pdz_denovo.generative.frames import centre

        trace = self._chains[i]
        start = 0
        if len(trace) > self.length:
            # Deterministic-ish crop; randomised by DataLoader shuffling of i.
            start = (i * 7919) % (len(trace) - self.length + 1)
        crop = trace[start : start + self.length]
        coords = torch.from_numpy(crop).float()
        return centre(coords)


def build_dataset(length: int, pdb_dir: str | Path | None = None, synthetic_n: int = 512):
    """Return a PDB dataset if data exists, else the synthetic fallback."""
    if pdb_dir is not None:
        ds = PDBBackboneDataset(pdb_dir, length=length)
        if len(ds) > 0:
            return ds
        LOGGER.warning("Using synthetic backbones (no PDB data available).")
    return SyntheticBackboneDataset(n=synthetic_n, length=length)
