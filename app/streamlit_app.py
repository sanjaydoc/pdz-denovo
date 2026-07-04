"""Streamlit dashboard for the pdz-denovo DBTL platform (Phase 6).

Reads the artifacts produced by the pipeline and presents them for a non-domain
expert ("model democratization"):

* pipeline status and target/config,
* DBTL optimization trajectory (hypervolume, Pareto size per cycle),
* the final Pareto front of designs (interactive 3D scatter + table),
* an optimizer benchmark (vs random search),
* a structure gallery of generated/validated backbones (py3Dmol).

Everything degrades gracefully: sections that have no data yet show a hint on how
to produce it. Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="PDZ De Novo — DBTL", layout="wide")


def _load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


# --- header ------------------------------------------------------------------
st.title("PDZ De Novo — Closed-Loop Binder Design")
st.caption("SE(3) flow matching + multi-objective optimization for PSD-95 PDZ binders")

cfg_path = ROOT / "configs" / "config.yaml"
try:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(cfg_path) if cfg_path.exists() else None
except Exception:  # noqa: BLE001
    cfg = None

if cfg is not None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Target", str(cfg.target.name))
    c2.metric("PDB ID", str(cfg.target.pdb_id))
    c3.metric("DBTL cycles", str(cfg.dbtl.n_cycles))
    c4.metric("Library size", str(cfg.dbtl.library_size))

phases = {
    "Phase 0 — Scaffold + data": "done",
    "Phase 1 — Oracle stack": "done",
    "Phase 2 — SE(3) flow-matching generator": "done",
    "Phase 3 — Sequence design (ProteinMPNN)": "done",
    "Phase 4 — ESMFold self-consistency": "done",
    "Phase 5 — DBTL loop + multi-objective optimization": "done",
    "Phase 6 — Dashboard + report": "done",
}
with st.expander("Pipeline status", expanded=False):
    for name, status in phases.items():
        st.write(f"- **{name}** — {status}")

# --- DBTL trajectory ---------------------------------------------------------
st.header("DBTL optimization trajectory")
history = _load_json(ROOT / "outputs" / "dbtl" / "history.json")
if not history:
    st.info("No DBTL run yet. Produce one with `python scripts/run_cycle.py`.")
else:
    import pandas as pd

    df = pd.DataFrame(history)
    a, b = st.columns(2)
    with a:
        st.subheader("Hypervolume per cycle")
        st.line_chart(df.set_index("cycle")["hypervolume"])
    with b:
        st.subheader("Pareto-front size per cycle")
        st.line_chart(df.set_index("cycle")["n_pareto"])
    st.caption(
        f"Final: {df['archive_size'].iloc[-1]} candidates evaluated, "
        f"{df['n_pareto'].iloc[-1]} on the Pareto front, "
        f"hypervolume {df['hypervolume'].iloc[-1]:.4f}."
    )

# --- Pareto front ------------------------------------------------------------
st.header("Final Pareto front")
pareto = _load_json(ROOT / "outputs" / "dbtl" / "pareto.json")
if not pareto:
    st.info("Run `python scripts/run_cycle.py` to populate the Pareto set.")
else:
    import pandas as pd

    pdf = pd.DataFrame(pareto)
    try:
        import plotly.express as px

        fig = px.scatter_3d(
            pdf, x="stability", y="solubility", z="binding",
            hover_data=["id", "sequence"], color="binding",
            title="Pareto-optimal designs (higher = better on every axis)",
        )
        fig.update_traces(marker=dict(size=5))
        st.plotly_chart(fig, use_container_width=True)
    except Exception:  # noqa: BLE001 - plotly optional
        st.scatter_chart(pdf, x="stability", y="solubility")
    st.dataframe(pdf, use_container_width=True)

# --- optimizer benchmark -----------------------------------------------------
st.header("Optimizer benchmark")
bench = _load_json(ROOT / "outputs" / "benchmark" / "benchmark.json")
if not bench:
    st.info("Run `python scripts/benchmark_optimizers.py` to compare optimizers vs random search.")
else:
    import pandas as pd

    cols = {name: r["hypervolume"] for name, r in bench.items()}
    max_len = max(len(v) for v in cols.values())
    data = {name: v + [None] * (max_len - len(v)) for name, v in cols.items()}
    bdf = pd.DataFrame(data)
    bdf.index.name = "cycle"
    st.subheader("Hypervolume vs cycle (higher/steeper = better optimizer)")
    st.line_chart(bdf)
    png = ROOT / "outputs" / "benchmark" / "benchmark.png"
    if png.exists():
        st.image(str(png))

# --- structure gallery -------------------------------------------------------
st.header("Structure gallery")
pdb_dirs = [ROOT / "outputs" / "validation", ROOT / "outputs" / "generator" / "samples"]
pdbs = [p for d in pdb_dirs if d.exists() for p in sorted(d.glob("*.pdb"))][:6]
if not pdbs:
    st.info("Generate backbones (`python scripts/validate_designs.py`) to view structures.")
else:
    try:
        import py3Dmol
        import streamlit.components.v1 as components

        cols = st.columns(min(3, len(pdbs)))
        for i, pdb_path in enumerate(pdbs):
            view = py3Dmol.view(width=320, height=260)
            view.addModel(pdb_path.read_text(), "pdb")
            view.setStyle({"cartoon": {"color": "spectrum"}})
            view.zoomTo()
            with cols[i % len(cols)]:
                st.caption(pdb_path.name)
                components.html(view._make_html(), height=280)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"py3Dmol view unavailable ({exc}). PDBs: {[p.name for p in pdbs]}")

st.divider()
st.caption(
    "Simulated wet-lab oracles (proxies, not validated assays). Design strategy: "
    "C-terminal presentation of the PDZ class-I motif. See docs/REPORT.md."
)
