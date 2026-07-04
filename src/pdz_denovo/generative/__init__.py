"""SE(3)-equivariant flow-matching backbone generator (Phase 2).

Importing this package requires PyTorch (the models subclass ``nn.Module``).
The rest of ``pdz_denovo`` does not import it eagerly, so the oracle stack and
utilities remain usable without a torch install.
"""
from pdz_denovo.generative.egnn import EGNNLayer, EGNNVelocityField, knn_graph
from pdz_denovo.generative.flow import FlowMatching, build_flow_model
from pdz_denovo.generative.frames import centre, kabsch_rmsd, random_rotation
from pdz_denovo.generative.sample import coords_to_pdb, sample_to_pdbs

__all__ = [
    "EGNNLayer",
    "EGNNVelocityField",
    "knn_graph",
    "FlowMatching",
    "build_flow_model",
    "centre",
    "kabsch_rmsd",
    "random_rotation",
    "coords_to_pdb",
    "sample_to_pdbs",
]
