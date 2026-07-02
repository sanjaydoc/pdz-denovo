# PDZ De Novo — Closed-Loop Protein Binder Design

A fully open-source **Design–Build–Test–Learn (DBTL)** platform for de novo design of
miniprotein binders against the **PSD-95 PDZ domain** (a central synaptic scaffolding
protein relevant to neural signaling and brain-computer interface tooling).

The system:

1. **Designs** candidate protein backbones with an SE(3)-equivariant **flow-matching**
   generative model (PyTorch).
2. **Assigns sequences** to backbones using an ESM-2 based inverse-folding / scoring step.
3. **Tests** candidates with a modular **simulated wet-lab oracle**:
   - Stability / foldability proxy (ESM-2 pseudo-log-likelihood; ESMFold pluggable).
   - Binding affinity proxy (AutoDock/`smina` docking to the PDZ domain).
   - Solubility / aggregation proxy (sequence-based model).
4. **Learns** by proposing the next library via **multi-objective optimization**
   (BoTorch qNEHVI or a genetic algorithm) over a Pareto front.
5. **Reports** progress in a **Streamlit** dashboard.

## Why this target?

PSD-95 PDZ domains organize the post-synaptic density and are fundamental to synaptic
signaling. A designed, modular binder is a plausible building block for targeting
BCI/neural-interface components to synapses — directly on-mission for neural-interface
research while remaining small enough to run on modest hardware.

## Hardware profile (dev machine)

- Intel i7-9850H (6C/12T), 16 GB RAM, 500 GB SSD
- NVIDIA Quadro RTX 3000, **6 GB VRAM**

The stack is tuned for 6 GB VRAM: small equivariant models, fp16, gradient
checkpointing, CPU fallbacks, and a lightweight default stability oracle.

## Quickstart

```powershell
# 1. Create the environment (one-time)
scripts\setup_env.ps1

# 2. Activate it
.\.venv\Scripts\Activate.ps1

# 3. Download the PDZ target structure + seed data
python scripts\download_data.py

# 4. Run a single DBTL cycle (once implemented)
python scripts\run_cycle.py

# 5. Launch the dashboard
streamlit run app\streamlit_app.py
```

## Project layout

```
pdz-denovo/
├── configs/            # Hydra configuration
├── src/pdz_denovo/     # Library code
│   ├── data/           # PDB loading, target prep, featurization
│   ├── generative/     # SE(3)-equivariant flow-matching model
│   ├── sequence/       # ESM-2 sequence design / scoring
│   ├── oracle/         # stability + docking + solubility oracles
│   ├── optimize/       # multi-objective optimization (BoTorch / GA)
│   ├── loop/           # DBTL orchestration
│   ├── tracking/       # MLflow + JSON/CSV logging
│   └── utils/          # config, logging, seeding
├── app/                # Streamlit dashboard
├── scripts/            # entrypoints (download_data, run_cycle, setup_env)
├── tests/              # pytest unit tests
└── notebooks/          # exploration + final report
```

## License / data notes

- Code: MIT (see `LICENSE`).
- ESM-2 weights: released by Meta under permissive research terms — verify before any
  commercial use.
- PDB structures: freely available from RCSB PDB.

## Status

Phase 0 (scaffold + data) — in progress. See milestone plan in project notes.
