"""SE(3)-equivariant EGNN velocity field for backbone flow matching.

This implements an **E(n) Equivariant Graph Neural Network** (Satorras et al.,
2021) specialised as the velocity field ``v_theta(x_t, t)`` of a flow-matching
model over Cα coordinates.

Why EGNN (rather than importing an equivariant library):
    The whole point of the portfolio is to *demonstrate* SE(3)-equivariance, so
    the equivariant coordinate update is written out explicitly. Messages depend
    only on **invariant** quantities (scalar node features and pairwise squared
    distances), while coordinates are updated along the **equivariant**
    relative-position vectors ``x_i - x_j``. Consequently:

        * a global rotation R of the input rotates the predicted velocity by R;
        * a global translation leaves the predicted velocity unchanged
          (messages use only differences).

    Both properties are asserted in ``tests/test_phase2_generative.py``.

The network is deliberately small (defaults from ``configs/model/flow.yaml``:
hidden 128, 5 layers, kNN 16, ≤80 residues) so it trains inside a 6 GB GPU with
fp16 autocast and optional gradient checkpointing.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

LOGGER = logging.getLogger("pdz_denovo")


def knn_graph(x: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the ``k`` nearest neighbours of each node (by Euclidean dist).

    Args:
        x: ``(B, L, 3)`` coordinates.
        k: number of neighbours (clamped to ``L - 1``).

    Returns:
        ``(B, L, k)`` long tensor of neighbour indices. Distances are invariant
        to rotation/translation, so the graph itself is SE(3)-invariant.
    """
    b, length, _ = x.shape
    k = min(k, length - 1)
    with torch.no_grad():
        dist2 = torch.cdist(x, x)  # (B, L, L)
        # Exclude self by pushing the diagonal to +inf.
        eye = torch.eye(length, device=x.device, dtype=torch.bool)
        dist2 = dist2.masked_fill(eye.unsqueeze(0), float("inf"))
        idx = dist2.topk(k, dim=-1, largest=False).indices  # (B, L, k)
    return idx


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal embedding of a scalar time ``t`` in ``[0, 1]``.

    Args:
        t: ``(B,)`` times.
        dim: embedding dimension (even).

    Returns:
        ``(B, dim)`` embedding.
    """
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(0.0, 1.0, half, device=t.device) * -torch.log(torch.tensor(10000.0))
    )
    args = t[:, None] * freqs[None, :] * 1000.0
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:  # pad if odd
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


class EGNNLayer(nn.Module):
    """One equivariant message-passing layer over a fixed kNN graph."""

    def __init__(self, hidden_dim: int, edge_dim: int = 32) -> None:
        super().__init__()
        # Edge/message network consumes [h_i, h_j, ||x_i - x_j||^2].
        self.edge_mlp = _mlp(2 * hidden_dim + 1, hidden_dim, edge_dim)
        # Scalar weight that scales each relative-position vector (equivariant).
        self.coord_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        # Node update network.
        self.node_mlp = _mlp(hidden_dim + edge_dim, hidden_dim, hidden_dim)
        self.edge_dim = edge_dim

    def forward(
        self, h: torch.Tensor, x: torch.Tensor, idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update node features ``h`` and coordinates ``x``.

        Args:
            h: ``(B, L, H)`` invariant node features.
            x: ``(B, L, 3)`` coordinates.
            idx: ``(B, L, k)`` neighbour indices.

        Returns:
            Updated ``(h, x)``.
        """
        b, length, k = idx.shape
        # Gather neighbour features/coords. The "key" tensors index residues on
        # dim=1 (unsqueeze(1)), so gathering along dim=2 selects each node's
        # actual neighbours; unsqueeze(2) would wrongly return the node itself.
        idx_h = idx.unsqueeze(-1).expand(b, length, k, h.shape[-1])
        h_j = torch.gather(h.unsqueeze(1).expand(b, length, length, h.shape[-1]), 2, idx_h)
        idx_x = idx.unsqueeze(-1).expand(b, length, k, 3)
        x_j = torch.gather(x.unsqueeze(1).expand(b, length, length, 3), 2, idx_x)

        h_i = h.unsqueeze(2).expand(b, length, k, h.shape[-1])
        rel = x.unsqueeze(2) - x_j  # (B, L, k, 3) equivariant
        dist2 = (rel**2).sum(dim=-1, keepdim=True)  # (B, L, k, 1) invariant

        # Messages depend only on invariants -> messages are invariant.
        m = self.edge_mlp(torch.cat([h_i, h_j, dist2], dim=-1))  # (B, L, k, E)

        # Equivariant coordinate update: weighted sum of relative vectors.
        # tanh bounds each per-neighbour contribution -> stable dynamics.
        coord_w = torch.tanh(self.coord_mlp(m))  # (B, L, k, 1) invariant scalar
        # Normalise relative vectors for numerical stability (unit direction).
        rel_unit = rel / (dist2.sqrt() + 1.0)
        dx = (rel_unit * coord_w).sum(dim=2) / k  # (B, L, 3) equivariant
        x = x + dx

        # Invariant node update from aggregated messages.
        m_agg = m.sum(dim=2)  # (B, L, E)
        h = h + self.node_mlp(torch.cat([h, m_agg], dim=-1))
        return h, x


class EGNNVelocityField(nn.Module):
    """Velocity field ``v_theta(x_t, t)`` for Cα flow matching.

    The forward pass returns a per-residue velocity vector that is
    **rotation-equivariant** and **translation-invariant** — exactly the gauge
    a flow over centred coordinates requires.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 5,
        edge_dim: int = 32,
        n_neighbors: int = 16,
        max_residues: int = 80,
        time_embed_dim: int = 32,
        grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.n_neighbors = n_neighbors
        self.grad_checkpoint = grad_checkpoint
        self.hidden_dim = hidden_dim

        # Invariant initial features: learned positional (sequence-index) embed
        # plus a projected time embedding. No absolute coordinates enter here,
        # which is what keeps the field translation-invariant.
        self.pos_embed = nn.Embedding(max_residues, hidden_dim)
        self.time_embed_dim = time_embed_dim
        self.time_mlp = _mlp(time_embed_dim, hidden_dim, hidden_dim)

        self.layers = nn.ModuleList(
            [EGNNLayer(hidden_dim, edge_dim=edge_dim) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the velocity at coordinates ``x`` and time ``t``.

        Args:
            x: ``(B, L, 3)`` (centred) coordinates.
            t: ``(B,)`` times in ``[0, 1]``.

        Returns:
            ``(B, L, 3)`` velocity vectors.
        """
        b, length, _ = x.shape
        if length > self.pos_embed.num_embeddings:
            raise ValueError(
                f"Sequence length {length} exceeds max_residues "
                f"{self.pos_embed.num_embeddings}."
            )

        pos = torch.arange(length, device=x.device)
        h = self.pos_embed(pos).unsqueeze(0).expand(b, length, self.hidden_dim).clone()
        temb = self.time_mlp(sinusoidal_time_embedding(t, self.time_embed_dim))
        h = h + temb.unsqueeze(1)  # broadcast time over residues

        idx = knn_graph(x, self.n_neighbors)
        x_in = x
        for layer in self.layers:
            if self.grad_checkpoint and self.training:
                h, x = torch.utils.checkpoint.checkpoint(
                    layer, h, x, idx, use_reentrant=False
                )
            else:
                h, x = layer(h, x, idx)

        # Velocity is the net equivariant displacement produced by the network.
        return x - x_in
