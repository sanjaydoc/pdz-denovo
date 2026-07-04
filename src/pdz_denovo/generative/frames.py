"""Backbone geometry helpers for the SE(3) flow-matching generator.

We represent a protein backbone by its Cα trace: a tensor of shape ``(L, 3)``
(or ``(B, L, 3)`` batched). Flow matching operates directly on these
coordinates, and an EGNN velocity field (see :mod:`egnn`) keeps the dynamics
**SE(3)-equivariant**: a global rotation/translation of the input produces the
same rotation/translation of the output.

This module holds the small, dependency-light geometry utilities used across
training, sampling, and validation:

* :func:`centre` — remove the centroid (translation gauge fixing).
* :func:`random_rotation` — sample a uniform rotation (used for the
  equivariance test and optional data augmentation).
* :func:`kabsch` / :func:`kabsch_rmsd` — optimal superposition and RMSD, the
  basis of the self-consistency (scRMSD) metric used later in validation.
"""
from __future__ import annotations


def centre(x, mask=None):
    """Subtract the (masked) centroid so coordinates are translation-free.

    Args:
        x: ``(..., L, 3)`` coordinates.
        mask: optional ``(..., L)`` boolean/float mask of valid residues.

    Returns:
        Centred coordinates with the same shape as ``x``.
    """
    import torch

    if mask is None:
        centroid = x.mean(dim=-2, keepdim=True)
    else:
        m = mask.unsqueeze(-1).to(x.dtype)  # (..., L, 1)
        denom = m.sum(dim=-2, keepdim=True).clamp_min(1.0)
        centroid = (x * m).sum(dim=-2, keepdim=True) / denom
    return x - centroid


def random_rotation(batch: int = 1, device=None, dtype=None):
    """Sample ``batch`` uniform rotation matrices via QR of a Gaussian matrix.

    Returns a tensor of shape ``(batch, 3, 3)`` with determinant +1.
    """
    import torch

    a = torch.randn(batch, 3, 3, device=device, dtype=dtype)
    q, r = torch.linalg.qr(a)
    # Make the decomposition unique / proper (det = +1).
    d = torch.diagonal(r, dim1=-2, dim2=-1).sign()
    q = q * d.unsqueeze(-2)
    det = torch.linalg.det(q)
    q[..., :, 0] = q[..., :, 0] * det.unsqueeze(-1)
    return q


def kabsch(P, Q):
    """Rotation matrix that optimally superposes ``P`` onto ``Q`` (per batch).

    Both inputs are ``(..., L, 3)`` and assumed already centred. Returns
    ``(..., 3, 3)`` rotation matrices ``R`` such that ``P @ R.transpose(-1,-2)``
    best matches ``Q`` in the least-squares sense.
    """
    import torch

    h = P.transpose(-1, -2) @ Q  # (..., 3, 3) covariance
    u, _, vt = torch.linalg.svd(h)
    d = torch.linalg.det(vt.transpose(-1, -2) @ u.transpose(-1, -2))
    # Correct for reflection to guarantee a proper rotation.
    diag = torch.ones_like(P[..., :3, 0])
    diag[..., -1] = d
    r = vt.transpose(-1, -2) @ torch.diag_embed(diag) @ u.transpose(-1, -2)
    return r


def kabsch_rmsd(P, Q):
    """Minimum RMSD between two point sets after optimal superposition.

    Args:
        P, Q: ``(..., L, 3)`` coordinate sets (need not be pre-centred).

    Returns:
        ``(...)`` RMSD values (a scalar for unbatched input).
    """
    import torch

    Pc = centre(P)
    Qc = centre(Q)
    r = kabsch(Pc, Qc)
    P_aligned = Pc @ r.transpose(-1, -2)
    diff = P_aligned - Qc
    msd = (diff**2).sum(dim=-1).mean(dim=-1)
    return torch.sqrt(msd.clamp_min(0.0))
