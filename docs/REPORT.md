# PDZ De Novo — Technical Report

A closed-loop **Design–Build–Test–Learn (DBTL)** platform for de novo design of
miniprotein binders against the **PSD-95 PDZ3 domain** (`1BE9`), a synaptic
scaffolding protein central to neural signaling. The project demonstrates an
end-to-end *closed-loop optimization backbone* — generative design, structure-
conditioned sequence design, in-silico validation, and multi-objective Bayesian /
evolutionary optimization — built to run on a single 6 GB laptop GPU.

## 1. Problem

Designing a protein binder is a search over an astronomically large space
(20^L sequences for length L). Physical assays are slow and expensive, so the
question is not "can we score one design" but "how do we *propose the few
candidates most likely to work, learn from results, and improve* — automatically."
That iterative loop is DBTL, and this platform is a working in-silico instance of
it.

## 2. Architecture

```
seed library
   │  (Design)          Phase 2: SE(3) flow-matching generator → Cα backbones
   │                    Phase 3: ProteinMPNN inverse folding   → sequences (+ PDZ motif)
   ▼
oracle scores           Phase 1: ESM-2 stability + solubility + PDZ binding
   │  (Test, in silico) Phase 4: ESMFold self-consistency (scRMSD / pLDDT)
   ▼
optimizer proposes      Phase 5: NSGA-II / BoTorch qNEHVI over the Pareto front
   │  (Learn)
   ▼
next library ─── repeat
```

The loop (`loop/dbtl.py`) is deliberately **component-agnostic**: it takes any
`scorer`, `optimizer`, and `seed_fn`. Swapping the oracle and target retargets
the whole machine — the "multiple biomolecular verticals" a general backbone
must support.

## 3. Methods

### 3.1 Generative model (Phase 2)
An **SE(3)-equivariant EGNN** velocity field trained with **conditional
(rectified) flow matching** over Cα coordinates. Messages depend only on
invariants (scalar features + pairwise squared distances); coordinates update
along the equivariant relative-position vectors, so the predicted velocity is
rotation-equivariant and translation-invariant (asserted in the test suite).
Coordinates are scaled to unit variance to match the Gaussian prior. Tuned for
6 GB: 128 hidden units, 5 layers, kNN-16 graph, fp16, gradient checkpointing.

### 3.2 Sequence design (Phase 3)
**ProteinMPNN** (Cα-only variant) assigns sequences to generated backbones —
structure-conditioned inverse folding, the field standard, and a transfer-
learning story (a pretrained model adapted to the task). The **PSD-95 class-I
motif** (`…S/T-X-Φ`) is grafted onto every design, encoding the domain prior for
a **C-terminal-presentation** binder strategy.

### 3.3 Simulated wet-lab oracle (Phase 1)
Three "higher-is-better" objectives: **stability** (ESM-2 pseudo-log-likelihood),
**solubility** (Kyte–Doolittle hydropathy + net charge + aggregation windows),
and **binding** (PSD-95 class-I motif compatibility). On-disk caching keyed by
sequence avoids re-scoring across cycles.

### 3.4 In-silico validation (Phase 4)
**ESMFold self-consistency**: designed backbone → ProteinMPNN → ESMFold refold →
**scRMSD** (Cα, Kabsch) and **pLDDT**. Success gate scRMSD < 2 Å and pLDDT > 70 —
the metric used across the de novo design literature. Folding runs on the
top-K designs only (the "expensive assay" spent on promising candidates), via an
API-first backend with a local fallback.

### 3.5 Optimization (Phase 5)
Two optimizers propose the next library from scored candidates:
- **NSGA-II** — non-dominated sort + crowding distance, tournament selection,
  crossover and mutation in sequence space (motif re-grafted each generation).
- **BoTorch qNEHVI** — batch Bayesian optimization: a GP surrogate over ESM-2
  embedding features selects the next library by q-Noisy Expected Hypervolume
  Improvement from a mutation-generated pool.

Progress is tracked as **hypervolume** (exact 2D/3D) and Pareto-front size over
the cumulative archive, logged to MLflow + JSON.

## 4. Results (illustrative, proxy oracles)

A representative 5-cycle run (random seed, NSGA-II, library 32):

| cycle | archive | Pareto | hypervolume | best stability | best solubility |
|------:|--------:|-------:|------------:|---------------:|----------------:|
| 0 | 16 | 5 | 0.0167 | −0.552 | 0.809 |
| 1 | 48 | 6 | 0.0220 | −0.516 | 0.828 |
| 2 | 80 | 10 | 0.0250 | −0.515 | 0.865 |
| 3 | 112 | 14 | 0.0267 | −0.515 | 0.865 |
| 4 | 144 | 17 | 0.0284 | −0.504 | 0.865 |

**Hypervolume rises monotonically** and the Pareto front grows — the loop is
learning and proposing better libraries. `scripts/benchmark_optimizers.py`
compares NSGA-II and qNEHVI against a random-search baseline on the same metric.

Observation: because the class-I motif is grafted onto every candidate, the
**binding objective saturates** near its maximum — it stops being the active
constraint, and the loop optimizes the stability/solubility trade-off. This is a
direct consequence of the design-prior choice, stated here for transparency.

## 5. Honest limitations

- The oracles are **cheap proxies**, not validated assays. A real campaign
  replaces them with physical assays or heavier models (AlphaFold + physics-based
  docking). The value demonstrated is the **architecture and closed-loop
  reasoning**, plus the self-consistency validation that mirrors the literature.
- The generator is trained at **toy scale** on a small set of short monomers;
  it demonstrates the method rather than state-of-the-art structures.
- Binding is a **motif proxy**, not full protein–protein docking.

## 6. Relevance to closed-loop molecular engineering

The platform is a miniature of a **closed-loop optimization backbone** that
connects experiments to ML: data ingestion, generative + surrogate modeling,
multi-objective/constrained library design, MLflow-tracked reproducibility, and a
dashboard for non-domain experts. The named techniques — SE(3)-equivariance,
flow matching, protein language models, transfer learning, multi-objective
Bayesian optimization — are each implemented and testable, and the components are
decoupled so the same backbone retargets to new verticals.

## 7. Reproducibility

Hydra configs, pinned `requirements.txt`, and pytest coverage of every stage
(equivariance, oracle behavior, motif logic, self-consistency math, Pareto /
hypervolume, and the closed loop). See `README.md` for setup and `docs/PLAN.md`
for the phase roadmap and hardware budget.

## References

EGNN (Satorras et al., 2021) · Flow Matching (Lipman et al., 2023) · Rectified
Flow (Liu et al., 2023) · FrameDiff / FrameFlow (Yim et al., 2023) · ProteinMPNN
(Dauparas et al., 2022) · ESM-2 / ESMFold (Lin et al., 2023) · qNEHVI
(Daulton et al., 2021) · NSGA-II (Deb et al., 2002).
