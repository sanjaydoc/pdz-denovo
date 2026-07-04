"""Streamlit dashboard for the pdz-denovo DBTL platform.

Phase 0: placeholder that shows project status and configuration. Later phases
populate this with optimization trajectories, Pareto fronts, and a structure
gallery.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omegaconf import OmegaConf  # noqa: E402

st.set_page_config(page_title="PDZ De Novo — DBTL", layout="wide")

st.title("PDZ De Novo — Closed-Loop Binder Design")
st.caption("SE(3) flow matching + multi-objective optimization for PSD-95 PDZ binders")

cfg_path = ROOT / "configs" / "config.yaml"
if cfg_path.exists():
    cfg = OmegaConf.load(cfg_path)
    col1, col2, col3 = st.columns(3)
    col1.metric("Target", cfg.target.name)
    col2.metric("PDB ID", cfg.target.pdb_id)
    col3.metric("DBTL cycles", cfg.dbtl.n_cycles)

    with st.expander("Full configuration"):
        st.code(OmegaConf.to_yaml(cfg), language="yaml")

st.info(
    "Phase 0 scaffold is live. Optimization trajectories, Pareto fronts, and the "
    "structure gallery will appear here as later phases are implemented."
)

st.subheader("Pipeline status")
phases = {
    "Phase 0 — Scaffold + data": "done",
    "Phase 1 — Oracle stack (ESM-2 + solubility + PDZ binding)": "done",
    "Phase 2 — SE(3) flow-matching generator": "done",
    "Phase 3 — Sequence design (inverse folding)": "pending",
    "Phase 4 — ESMFold self-consistency validation": "pending",
    "Phase 5 — DBTL loop + multi-objective optimization": "pending",
    "Phase 6 — Dashboard + report": "pending",
}
for name, status in phases.items():
    st.write(f"- **{name}** — {status}")
