"""Experiment tracking for DBTL runs — MLflow with a JSON fallback.

Logs per-cycle metrics (hypervolume, Pareto size, best-per-objective) so a run
is reproducible and inspectable. MLflow is used when available (the intended,
fully-open-source stack); otherwise everything still lands in a JSON file so the
loop never depends on a tracking server being up.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

LOGGER = logging.getLogger("pdz_denovo")


class RunTracker:
    def __init__(
        self,
        out_dir: str | Path = "outputs/dbtl",
        experiment: str = "pdz-denovo-dbtl",
        use_mlflow: bool = True,
        params: dict | None = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cycles: list[dict] = []
        self._mlflow = None
        if use_mlflow:
            try:
                import mlflow

                self._mlflow = mlflow
                mlflow.set_experiment(experiment)
                mlflow.start_run()
                if params:
                    mlflow.log_params(params)
            except Exception:  # noqa: BLE001 - tracking is optional
                LOGGER.info("MLflow unavailable; logging DBTL metrics to JSON only.")
                self._mlflow = None
        self._params = params or {}

    def log_cycle(self, metrics: dict) -> None:
        self.cycles.append(metrics)
        if self._mlflow is not None:
            step = metrics.get("cycle", len(self.cycles) - 1)
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    self._mlflow.log_metric(key, val, step=step)
            self._mlflow.log_metric("best_stability", metrics["best_per_objective"][0], step=step)
        # Persist incrementally so a crash still leaves a trail.
        (self.out_dir / "history.json").write_text(json.dumps(self.cycles, indent=2))

    def finish(self, summary: dict) -> None:
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        if self._mlflow is not None:
            self._mlflow.log_artifact(str(self.out_dir / "history.json"))
            self._mlflow.end_run()
