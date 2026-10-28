"""AutoML adapter layer — contracts and subprocess runner infrastructure.

Each framework runner:
  - Receives a serialised TrainSpec (JSON) on stdin
  - Writes a RunResult (JSON) to stdout
  - Runs in its isolated .envs/<framework_id>/ virtualenv

This module is the *host-side* orchestrator. It:
  1. Serialises the TrainSpec.
  2. Spawns the correct venv python with the per-framework worker script.
  3. Deserialises + returns the RunResult.

Framework workers live in src/acbs/automl/workers/<framework_id>_worker.py
and are injected as -c code to avoid import-path issues.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data-transfer objects
# ---------------------------------------------------------------------------


@dataclass
class TrainSpec:
    """Serialisable specification sent to each framework worker."""

    framework_id: str
    X_train: list[list[Any]]
    y_train: list[Any]
    X_test: list[list[Any]]
    feature_names: list[str]
    task_type: str  # "binary" | "multiclass"
    n_classes: int
    time_budget_s: int
    seed: int = 42
    metric: str = "roc_auc"

    def to_json(self) -> str:
        return json.dumps(
            {
                "framework_id": self.framework_id,
                "X_train": self.X_train,
                "y_train": self.y_train,
                "X_test": self.X_test,
                "feature_names": self.feature_names,
                "task_type": self.task_type,
                "n_classes": self.n_classes,
                "time_budget_s": self.time_budget_s,
                "seed": self.seed,
                "metric": self.metric,
            }
        )

    @classmethod
    def from_arrays(
        cls,
        framework_id: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        task_type: str,
        n_classes: int,
        time_budget_s: int,
        seed: int = 42,
        metric: str = "roc_auc",
    ) -> TrainSpec:
        return cls(
            framework_id=framework_id,
            X_train=X_train.values.tolist(),
            y_train=y_train.tolist(),
            X_test=X_test.values.tolist(),
            feature_names=list(X_train.columns),
            task_type=task_type,
            n_classes=n_classes,
            time_budget_s=time_budget_s,
            seed=seed,
            metric=metric,
        )


@dataclass
class RunResult:
    """Result returned from a framework worker subprocess."""

    framework_id: str
    status: str  # "ok" | "failed" | "timeout" | "skipped" | "unsupported"
    proba: list[list[float]] | None = None  # shape (n_test, n_classes)
    fit_time_s: float = 0.0
    predict_time_s: float = 0.0
    wall_time_s: float = 0.0
    framework_version: str | None = None
    best_pipeline: str | None = None  # human-readable description
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def proba_array(self) -> np.ndarray | None:
        if self.proba is None:
            return None
        return np.array(self.proba)

    @classmethod
    def from_json(cls, framework_id: str, raw: str) -> RunResult:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            return cls(
                framework_id=framework_id, status="failed", error=f"JSON decode error: {exc}"
            )
        return cls(
            framework_id=framework_id,
            status=data.get("status", "failed"),
            proba=data.get("proba"),
            fit_time_s=data.get("fit_time_s", 0.0),
            predict_time_s=data.get("predict_time_s", 0.0),
            wall_time_s=data.get("wall_time_s", 0.0),
            framework_version=data.get("framework_version"),
            best_pipeline=data.get("best_pipeline"),
            error=data.get("error"),
            extra=data.get("extra", {}),
        )

    @classmethod
    def unsupported(cls, framework_id: str, reason: str) -> RunResult:
        return cls(framework_id=framework_id, status="unsupported", error=reason)

    @classmethod
    def failed(cls, framework_id: str, reason: str, wall_time_s: float = 0.0) -> RunResult:
        return cls(
            framework_id=framework_id, status="failed", error=reason, wall_time_s=wall_time_s
        )

    @classmethod
    def timeout(cls, framework_id: str, budget_s: int) -> RunResult:
        return cls(
            framework_id=framework_id,
            status="timeout",
            error=f"Worker timed out after {budget_s + 60}s",
            wall_time_s=float(budget_s + 60),
        )
