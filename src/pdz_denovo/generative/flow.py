"""Conditional flow matching over backbone Cα coordinates.

We use **rectified / conditional-OT flow matching** (Lipman et al., 2023;
Liu et al., 2023): the simplest, most stable variant and a strong fit for a
6 GB budget.

Construction:
    * ``x1`` — a real (centred) backbone drawn from data.
    * ``x0`` — Gaussian noise, centred.
    * linear interpolation path  ``x_t = (1 - t) x0 + t x1``  for ``t ~ U(0, 1)``.
    * the target velocity along this path is the **constant** ``u = x1 - x0``.

The model learns ``v_theta(x_t, t) ≈ x1 - x0`` under an MSE loss. Sampling
integrates the learned ODE ``dx/dt = v_theta(x, t)`` from noise (``t = 0``) to a
designed backbone (``t = 1``).

Everything is done on **centred** coordinates so translations are gauged out;
the EGNN field supplies rotation-equivariance. Together this makes the whole
generative process SE(3)-equivariant.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from pdz_denovo.generative.egnn import EGNNVelocityField
from pdz_denovo.generative.frames import centre


class FlowMatching(nn.Module):
    """Wraps a velocity field with the flow-matching loss and an ODE sampler.

    Args:
        field: the SE(3)-equivariant velocity field.
        coord_scale: Cα coordinates (Å) are divided by this before flow matching
            so that real backbones have roughly unit variance and therefore match
            the ``N(0, I)`` noise prior. Sampling multiplies back by it. A backbone
            with radius of gyration ~10–15 Å is well served by the default. Getting
            this scale right is essential: raw-Å data vs unit-variance noise is a
            distribution mismatch that destabilises training.
    """

    def __init__(self, field: EGNNVelocityField, coord_scale: float = 10.0) -> None:
        super().__init__()
        self.field = field
        self.coord_scale = float(coord_scale)

    # --- training ------------------------------------------------------------
    @staticmethod
    def interpolate(
        x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(x_t, target_velocity)`` for the rectified-flow path.

        Args:
            x0: ``(B, L, 3)`` noise endpoint (t = 0).
            x1: ``(B, L, 3)`` data endpoint (t = 1).
            t: ``(B,)`` interpolation times.

        Returns:
            ``x_t`` and the constant target velocity ``x1 - x0``.
        """
        tt = t.view(-1, 1, 1)
        x_t = (1.0 - tt) * x0 + tt * x1
        target = x1 - x0
        return x_t, target

    def loss(self, x1: torch.Tensor) -> torch.Tensor:
        """Flow-matching MSE loss for a batch of real backbones ``x1``.

        Args:
            x1: ``(B, L, 3)`` backbone Cα coordinates (will be centred).

        Returns:
            Scalar loss.
        """
        x1 = centre(x1) / self.coord_scale
        x0 = centre(torch.randn_like(x1))
        t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
        x_t, target = self.interpolate(x0, x1, t)
        pred = self.field(x_t, t)
        return ((pred - target) ** 2).sum(dim=-1).mean()

    # --- sampling ------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        length: int,
        n_steps: int = 100,
        device=None,
        dtype=None,
    ) -> torch.Tensor:
        """Generate backbones by integrating the learned ODE from noise.

        Uses explicit Euler integration of ``dx/dt = v_theta(x, t)`` over
        ``n_steps`` uniform steps from ``t = 0`` to ``t = 1``.

        Returns:
            ``(n_samples, length, 3)`` designed Cα coordinates (centred).
        """
        self.eval()
        param = next(self.parameters())
        device = device or param.device
        dtype = dtype or param.dtype
        x = centre(torch.randn(n_samples, length, 3, device=device, dtype=dtype))
        dt = 1.0 / n_steps
        for step in range(n_steps):
            t = torch.full((n_samples,), step * dt, device=device, dtype=dtype)
            v = self.field(x, t)
            x = centre(x + v * dt)
        # Map back from unit-variance space to Ångström coordinates.
        return x * self.coord_scale


def build_flow_model(cfg) -> FlowMatching:
    """Construct a :class:`FlowMatching` model from a flow config node.

    Args:
        cfg: an OmegaConf node mirroring ``configs/model/flow.yaml``.
    """
    field = EGNNVelocityField(
        hidden_dim=int(cfg.hidden_dim),
        n_layers=int(cfg.n_layers),
        edge_dim=int(cfg.edge_dim),
        n_neighbors=int(cfg.n_neighbors),
        max_residues=int(cfg.max_residues),
        time_embed_dim=int(cfg.time_embed_dim),
        grad_checkpoint=bool(cfg.get("grad_checkpoint", False)),
    )
    return FlowMatching(field, coord_scale=float(cfg.get("coord_scale", 10.0)))
