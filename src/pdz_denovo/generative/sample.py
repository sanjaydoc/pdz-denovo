"""Sampling utilities: generate backbones and write them as Cα PDB traces.

The heavy lifting lives in :meth:`FlowMatching.sample`; this module adds a tiny
dependency-free PDB writer so sampled backbones can be inspected in PyMOL /
py3Dmol or handed to the sequence-design stage (Phase 3).
"""
from __future__ import annotations

import logging
from pathlib import Path

LOGGER = logging.getLogger("pdz_denovo")

_PDB_ATOM = (
    "ATOM  {serial:>5d}  CA  ALA {chain}{resseq:>4d}    "
    "{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C\n"
)


def coords_to_pdb(coords, path: str | Path, chain: str = "A") -> Path:
    """Write an ``(L, 3)`` Cα trace to a minimal PDB file (poly-alanine).

    Args:
        coords: ``(L, 3)`` array/tensor of Cα coordinates.
        path: output ``.pdb`` path.
        chain: chain identifier.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Accept torch tensors or numpy arrays without importing either eagerly.
    coords = [[float(v) for v in row] for row in coords]
    lines = []
    for i, (x, y, z) in enumerate(coords, start=1):
        lines.append(
            _PDB_ATOM.format(serial=i, chain=chain, resseq=i, x=x, y=y, z=z)
        )
    lines.append("TER\nEND\n")
    path.write_text("".join(lines))
    return path


def sample_to_pdbs(model, n_samples: int, length: int, out_dir: str | Path, n_steps: int = 100):
    """Sample ``n_samples`` backbones and write one PDB each.

    Returns the list of written paths.
    """
    out_dir = Path(out_dir)
    coords = model.sample(n_samples=n_samples, length=length, n_steps=n_steps)
    paths = []
    for i in range(coords.shape[0]):
        p = coords_to_pdb(coords[i].cpu(), out_dir / f"design_{i:03d}.pdb")
        paths.append(p)
    LOGGER.info("Wrote %d sampled backbones to %s.", len(paths), out_dir)
    return paths
