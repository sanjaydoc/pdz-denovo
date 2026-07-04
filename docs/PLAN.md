# PDZ De Novo — Project Roadmap

A Design–Build–Test–Learn (DBTL) platform for de novo design of miniprotein
binders against the **PSD-95 PDZ3 domain** (`1BE9`). The project is a working
miniature of a **closed-loop optimization backbone**: a generative model
proposes candidates, a simulated wet-lab oracle scores them, and a multi-objective
optimizer proposes the next, better library — iterating toward the Pareto front.

## Design philosophy

The framework is deliberately **target-agnostic**: the generator, oracle,
optimizer, and loop sit behind clean interfaces so PSD-95 is just the demo
vertical. Swapping the oracle and target repoints the whole machine at enzymes,
antibodies, or other biomolecular tools.

### Key design decision — binder strategy

The binding oracle scores whether a design's **own C-terminus** matches the
PSD-95 **class-I motif** `...[S/T]-X-Φ-COOH`. This is only meaningful under a
deliberate **C-terminal-presentation** strategy: the miniprotein engages the
PDZ groove by presenting an S/T-X-Φ tail (a well-precedented PDZ-binder mode).
Stated explicitly here so the assumption is auditable.

### Honest caveat

The oracles are cheap proxies, not validated assays. The value is the
**architecture and closed-loop thinking**, plus in-silico self-consistency
validation (Phase 4) that mirrors how the de novo design literature reports
success. A real campaign would swap the proxies for physical assays or heavier
models (AlphaFold/physics-based docking).

## Phases

| Phase | Module | Status | Deliverable |
|-------|--------|--------|-------------|
| 0 | `configs/`, `scripts/`, `utils/` | ✅ done | Hydra config tree, RCSB data download, env setup, logging/seed/device utils, tests |
| 1 | `oracle/` | ✅ done | ESM-2 stability (pseudo-log-likelihood) + biophysical solubility + PDZ class-I binding, combined `OracleStack` with disk caching |
| 2 | `generative/` | ✅ done | SE(3)-equivariant **EGNN velocity field** + **conditional flow matching** over Cα coords; ODE sampler; equivariance test suite |
| 3 | `sequence/` | ⏳ next | Inverse folding — **ProteinMPNN** assigns sequences to backbones, biased toward the C-terminal PDZ motif; ESM-2 secondary scoring |
| 4 | `fold/` | ⏳ planned | **ESMFold self-consistency**: design → ProteinMPNN → refold → **scRMSD** + **pLDDT/pTM**. Top-K per cycle only |
| 5 | `loop/`, `optimize/`, `tracking/` | ⏳ planned | DBTL orchestrator + **BoTorch qNEHVI** multi-objective BO (NSGA-II fallback) over an ESM-2 latent surrogate; MLflow tracking |
| 6 | `app/`, `notebooks/` | ⏳ planned | Live Streamlit (trajectories, Pareto front, py3Dmol gallery) + benchmark/ablation report (BO vs random vs GA) |

## Phase 2 detail (implemented)

* `frames.py` — centring, uniform random rotation, Kabsch superposition / RMSD
  (the basis of the scRMSD metric used in Phase 4).
* `egnn.py` — `EGNNVelocityField`: messages depend only on invariants (scalar
  features + pairwise squared distances); coordinates update along the
  equivariant relative-position vectors `x_i - x_j`. Result: velocity is
  **rotation-equivariant** and **translation-invariant**.
* `flow.py` — rectified / conditional-OT flow matching. Path
  `x_t = (1-t) x0 + t x1`, constant target velocity `x1 - x0`, MSE loss; Euler
  ODE sampler from noise → backbone.
* `dataset.py` — real PDB Cα crops (`PDBBackboneDataset`) with a synthetic
  α-helix fallback (`SyntheticBackboneDataset`) so the loop/tests run before any
  data download.
* `train.py` / `scripts/train_generator.py` — fp16 autocast, gradient
  checkpointing, grad clipping, MLflow + JSON logging.

References: EGNN (Satorras et al., 2021); Flow Matching (Lipman et al., 2023);
Rectified Flow (Liu et al., 2023); FrameDiff/FrameFlow; ProteinMPNN (Dauparas et
al., 2022); ESMFold (Lin et al., 2023); qNEHVI (Daulton et al., 2021).

## Hardware budget (RTX 3000 Mobile, 6 GB VRAM / 16 GB RAM)

| Phase | Fits? | How |
|-------|-------|-----|
| 2 — flow matching | ✅ | 128 hidden / 5 layers / kNN 16 / ≤80 res / batch 4 / fp16 / grad-checkpoint → ~2–4 GB VRAM |
| 3 — ProteinMPNN | ✅ | ~1.6M params, runs on CPU |
| 4 — ESMFold | ⚠️ | Weights ~5.6 GB fp16 — do **not** put on the 6 GB GPU or in the hot loop. Fold **top-K only** via the ESMFold public API (no local memory), CPU / OmegaFold fallback |
| 5 — qNEHVI BO | ✅ | GP over a few dozen points; ESM-2 35M embeddings as features (<2 GB) |
| 6 — dashboard | ✅ | CPU / browser |

**Model-size ladder (pinned to VRAM):** loop scoring / BO surrogate →
ESM-2 35M; optional final re-rank → ESM-2 150M; never 650M during the live loop.
Run generator training, ESMFold validation, and the BO loop as **separate
processes** — never one Python session holding all three (16 GB RAM limit).
