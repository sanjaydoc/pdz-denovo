"""Phase 2 tests for the SE(3) flow-matching generator.

The load-bearing test is **equivariance**: rotating the input must rotate the
predicted velocity identically, and translating the input must leave it
unchanged. These are the properties that justify calling the model
"SE(3)-equivariant" at all.

All tests are skipped cleanly when PyTorch is not installed, so the lightweight
Phase 0/1 suite still runs on a bare machine.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pdz_denovo.generative.egnn import EGNNVelocityField, knn_graph  # noqa: E402
from pdz_denovo.generative.flow import FlowMatching  # noqa: E402
from pdz_denovo.generative.frames import (  # noqa: E402
    centre,
    kabsch_rmsd,
    random_rotation,
)


def _small_field(seed: int = 0) -> EGNNVelocityField:
    torch.manual_seed(seed)
    # Use float64 so the equivariance check is tight (numerical, not statistical).
    field = EGNNVelocityField(
        hidden_dim=32, n_layers=3, edge_dim=16, n_neighbors=8, max_residues=32
    ).double()
    field.eval()
    return field


def test_velocity_shape():
    field = _small_field()
    x = torch.randn(2, 16, 3, dtype=torch.float64)
    t = torch.rand(2, dtype=torch.float64)
    v = field(x, t)
    assert v.shape == (2, 16, 3)


def test_rotation_equivariance():
    field = _small_field()
    x = centre(torch.randn(2, 16, 3, dtype=torch.float64))
    t = torch.rand(2, dtype=torch.float64)
    r = random_rotation(1, dtype=torch.float64)[0]

    v = field(x, t)
    v_rot = field(x @ r.transpose(-1, -2), t)
    # field(R x) should equal R (field(x)).
    assert torch.allclose(v_rot, v @ r.transpose(-1, -2), atol=1e-6, rtol=1e-5)


def test_translation_invariance():
    field = _small_field()
    x = centre(torch.randn(2, 16, 3, dtype=torch.float64))
    t = torch.rand(2, dtype=torch.float64)
    shift = torch.tensor([3.0, -1.0, 2.0], dtype=torch.float64)

    v = field(x, t)
    v_shift = field(x + shift, t)
    assert torch.allclose(v_shift, v, atol=1e-8)


def test_knn_graph_excludes_self_and_shape():
    x = torch.randn(1, 10, 3)
    idx = knn_graph(x, k=4)
    assert idx.shape == (1, 10, 4)
    # No node is its own neighbour.
    self_idx = torch.arange(10).view(1, 10, 1)
    assert not (idx == self_idx).any()


def test_flow_loss_is_scalar_and_differentiable():
    field = EGNNVelocityField(hidden_dim=32, n_layers=2, edge_dim=16, n_neighbors=8)
    model = FlowMatching(field)
    x1 = torch.randn(3, 16, 3)
    loss = model.loss(x1)
    assert loss.ndim == 0 and loss.item() >= 0.0
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_interpolate_endpoints():
    x0 = torch.randn(2, 8, 3)
    x1 = torch.randn(2, 8, 3)
    xt0, target = FlowMatching.interpolate(x0, x1, torch.zeros(2))
    xt1, _ = FlowMatching.interpolate(x0, x1, torch.ones(2))
    assert torch.allclose(xt0, x0, atol=1e-6)
    assert torch.allclose(xt1, x1, atol=1e-6)
    assert torch.allclose(target, x1 - x0, atol=1e-6)


def test_sampler_shape_and_centred():
    field = EGNNVelocityField(hidden_dim=16, n_layers=2, edge_dim=8, n_neighbors=6)
    model = FlowMatching(field)
    coords = model.sample(n_samples=2, length=12, n_steps=5)
    assert coords.shape == (2, 12, 3)
    # Output should be (approximately) centred.
    assert torch.allclose(coords.mean(dim=1), torch.zeros(2, 3), atol=1e-4)


def test_kabsch_rmsd_zero_for_rotated_copy():
    x = centre(torch.randn(1, 20, 3, dtype=torch.float64))
    r = random_rotation(1, dtype=torch.float64)[0]
    rmsd = kabsch_rmsd(x @ r.transpose(-1, -2), x)
    assert rmsd.item() < 1e-6


def test_synthetic_dataset_shapes():
    from pdz_denovo.generative.dataset import SyntheticBackboneDataset

    ds = SyntheticBackboneDataset(n=8, length=24)
    assert len(ds) == 8
    sample = ds[0]
    assert sample.shape == (24, 3)
    # Centred by construction.
    assert torch.allclose(sample.mean(dim=0), torch.zeros(3), atol=1e-4)


def test_coords_to_pdb_writes_file(tmp_path):
    from pdz_denovo.generative.sample import coords_to_pdb

    coords = torch.randn(10, 3)
    out = coords_to_pdb(coords, tmp_path / "design.pdb")
    text = out.read_text()
    assert out.exists()
    assert text.count("ATOM") == 10
    assert text.strip().endswith("END")
