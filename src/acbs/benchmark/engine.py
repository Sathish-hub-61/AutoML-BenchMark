from __future__ import annotations

import importlib.metadata
import time
import traceback
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.model_selection import cross_validate

from acbs.benchmark.metrics import compute_metrics, get_scorers, predict_proba_or_decision
from acbs.benchmark.timing import measure_prediction_time, measure_training_time
from acbs.benchmark.tuner import tune_model
from acbs.data.loader import LoadedDataset
from acbs.data.profiler import DatasetProfile, profile_dataset
from acbs.data.splitter import make_cv_splitter, make_holdout_split
from acbs.models.registry import ModelSpec, apply_class_weight, get_models
from acbs.preprocessing.imbalance import get_imbalance_strategy
from acbs.preprocessing.pipeline import build_full_pipeline, build_preprocessor
from acbs.utils.logging import get_logger
from acbs.utils.paths import load_default_config
from acbs.utils.seeds import set_global_seed

logger = get_logger(__name__)
ProgressCb = Callable[[str, int, int, str], None] | None


class BenchmarkConfig(BaseModel):
    models: list[str] = Field(default_factory=lambda: ["all"])
    cv_folds: int = 5
    seed: int = 42
    metrics_primary: str = "f1_macro"
    tuning: str = "none"
    n_trials: int = 20
    time_budget_seconds: int = 120
    total_budget_seconds: int = 1800
    n_jobs: int = 1
    imbalance_strategy: str = "auto"
    test_size: float = 0.2


class ModelRunResult(BaseModel):
    model_name: str
    status: str
    error: str | None = None
    notes: str = ""
    cv_metrics: dict[str, float] = Field(default_factory=dict)
    cv_std: dict[str, float] = Field(default_factory=dict)
    test_metrics: dict[str, float] = Field(default_factory=dict)
    fold_scores: dict[str, list[float]] = Field(default_factory=dict)
    training_time: float | None = None
    prediction_time_per_1k: float | None = None
    best_params: dict[str, Any] = Field(default_factory=dict)
    interpretability_score: int = 3
    y_pred: list[int] = Field(default_factory=list)
    y_true: list[int] = Field(default_factory=list)
    y_proba: list[list[float]] | None = None


class BenchmarkResult(BaseModel):
    profile: dict[str, Any]
    config: dict[str, Any]
    seed: int
    n_jobs: int
    lib_versions: dict[str, str]
    models: list[ModelRunResult]
    classes: list[str]


def library_versions() -> dict[str, str]:
    out = {}
    for pkg in ["scikit-learn", "xgboost", "pandas", "numpy", "optuna"]:
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except Exception:
            out[pkg] = "unknown"
    return out


def _cv_metrics_from_sklearn(cv_result: dict) -> tuple[dict[str, float], dict[str, float], dict[str, list[float]]]:
    means, stds, folds = {}, {}, {}
    mapping = {
        "test_accuracy": "accuracy",
        "test_f1_macro": "f1_macro",
        "test_precision_macro": "precision_macro",
        "test_recall_macro": "recall_macro",
        "test_balanced_accuracy": "balanced_accuracy",
        "test_roc_auc_ovr": "roc_auc",
        "fit_time": "fit_time",
    }
    for key, name in mapping.items():
        if key in cv_result:
            arr = [float(x) for x in cv_result[key]]
            folds[name] = arr
            means[name] = float(np.mean(arr))
            stds[name] = float(np.std(arr))
    return means, stds, folds


def _run_one(
    spec: ModelSpec,
    X_train,
    y_train,
    X_test,
    y_test,
    profile: DatasetProfile,
    cfg: dict,
    cv,
    classes,
) -> ModelRunResult:
    n_rows = len(X_train) + len(X_test)
    notes = []
    if spec.max_rows and n_rows > spec.max_rows:
        return ModelRunResult(
            model_name=spec.name,
            status="skipped",
            notes=f"Skipped: {n_rows} rows exceed max_rows={spec.max_rows}",
            interpretability_score=spec.interpretability_score,
        )
    imbalance = get_imbalance_strategy(profile, cfg, cfg.get("imbalance_strategy"))
    preprocessor = build_preprocessor(profile, cfg, spec.family, X_train)
    timeout = float(cfg.get("time_budget_seconds", 120))
    seed = int(cfg["seed"])
    n_jobs = int(cfg.get("n_jobs", 1))
    try:
        best_params, _history = tune_model(spec, X_train, y_train, preprocessor, cfg, cv, imbalance)
        est = spec.estimator_factory(best_params, seed, n_jobs)
        apply_class_weight(est, imbalance, y_train)
        pipe = build_full_pipeline(est, preprocessor, imbalance)
        scorers = get_scorers()
        cv_result = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=scorers,
            return_train_score=False,
            n_jobs=n_jobs,
        )
        means, stds, folds = _cv_metrics_from_sklearn(cv_result)
        with measure_training_time() as tinfo:
            pipe.fit(X_train, y_train)
        pred_t = measure_prediction_time(pipe, X_test)
        y_pred = pipe.predict(X_test)
        y_proba = predict_proba_or_decision(pipe, X_test)
        test_metrics = compute_metrics(y_test, y_pred, y_proba, classes)
        return ModelRunResult(
            model_name=spec.name,
            status="ok",
            notes="; ".join(notes),
            cv_metrics=means,
            cv_std=stds,
            test_metrics=test_metrics,
            fold_scores=folds,
            training_time=float(tinfo["seconds"]),
            prediction_time_per_1k=pred_t,
            best_params=best_params,
            interpretability_score=spec.interpretability_score,
            y_pred=[int(x) for x in np.asarray(y_pred).tolist()],
            y_true=[int(x) for x in np.asarray(y_test).tolist()],
            y_proba=y_proba.tolist() if y_proba is not None else None,
        )
    except TimeoutError as exc:
        logger.warning("timeout %s: %s", spec.name, exc)
        return ModelRunResult(model_name=spec.name, status="timeout", error=str(exc), notes=str(timeout))
    except Exception as exc:
        logger.exception("model failed %s", spec.name)
        return ModelRunResult(
            model_name=spec.name,
            status="failed",
            error=f"{exc}\n{traceback.format_exc(limit=4)}",
            interpretability_score=spec.interpretability_score,
        )


def run_benchmark(
    dataset: LoadedDataset,
    cfg: dict | None = None,
    progress_callback: ProgressCb = None,
) -> BenchmarkResult:
    """Train models sequentially so timings stay comparable; folds may use n_jobs."""
    base = load_default_config()
    if cfg:
        base.update({k: v for k, v in cfg.items() if v is not None})
    parsed = BenchmarkConfig.model_validate(
        {k: v for k, v in base.items() if k in BenchmarkConfig.model_fields}
    )
    merged = {**base, **parsed.model_dump()}
    set_global_seed(parsed.seed)
    profile = profile_dataset(
        dataset.X,
        dataset.y,
        imbalance_threshold=float(base.get("imbalance_ratio_threshold", 3)),
        high_cardinality=int(base.get("high_cardinality", 15)),
        id_unique_ratio=float(base.get("id_unique_ratio", 0.95)),
    )
    X_train, X_test, y_train, y_test = make_holdout_split(
        dataset.X, dataset.y, test_size=parsed.test_size, seed=parsed.seed
    )
    cv = make_cv_splitter(cv_folds=parsed.cv_folds, seed=parsed.seed)
    specs = get_models(None if parsed.models == ["all"] else parsed.models)
    classes = list(range(int(pd.Series(dataset.y).nunique())))
    results: list[ModelRunResult] = []
    deadline = time.perf_counter() + float(parsed.total_budget_seconds)
    for i, spec in enumerate(specs, start=1):
        if progress_callback:
            progress_callback(spec.name, i - 1, len(specs), f"Training {spec.name}")
        if time.perf_counter() > deadline:
            results.append(
                ModelRunResult(model_name=spec.name, status="timeout", error="Total time budget exhausted")
            )
            continue
        results.append(
            _run_one(spec, X_train, y_train, X_test, y_test, profile, merged, cv, classes)
        )
        if progress_callback:
            progress_callback(spec.name, i, len(specs), f"Finished {spec.name}")
    return BenchmarkResult(
        profile=profile.model_dump(),
        config=merged,
        seed=parsed.seed,
        n_jobs=parsed.n_jobs,
        lib_versions=library_versions(),
        models=results,
        classes=list(dataset.meta.get("classes", [str(c) for c in classes])),
    )
