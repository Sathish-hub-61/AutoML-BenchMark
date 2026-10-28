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

from acbs.automl.contracts import TrainSpec
from acbs.automl.evaluate import evaluate_automl_result
from acbs.automl.runner import run_framework
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
# Callback signature: (model_name, step, total, message, status=None, elapsed_s=None) -> None
ProgressCb = Callable[..., None] | None


class BenchmarkConfig(BaseModel):
    models: list[str] = Field(default_factory=lambda: ["all"])
    automl_frameworks: list[str] = Field(default_factory=list)
    automl_budget_s: int = 60
    automl_n_jobs: int = 1
    automl_grace_s: int = 60
    automl_memory_gb: int = 4
    outer_folds: int = 1
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
    kind: str = "classic"
    framework: str | None = None
    framework_version: str | None = None
    budget_s: int | None = None
    budget_used_ratio: float | None = None
    n_models_evaluated: int | None = None
    model_summary: str | None = None
    peak_mem_mb: float | None = None


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


def _cv_metrics_from_sklearn(
    cv_result: dict,
) -> tuple[dict[str, float], dict[str, float], dict[str, list[float]]]:
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
        return ModelRunResult(
            model_name=spec.name, status="timeout", error=str(exc), notes=str(timeout)
        )
    except Exception as exc:
        logger.exception("model failed %s", spec.name)
        return ModelRunResult(
            model_name=spec.name,
            status="failed",
            error=f"{exc}\n{traceback.format_exc(limit=4)}",
            interpretability_score=spec.interpretability_score,
        )


def estimate_cost(cfg: BenchmarkConfig | dict[str, Any], n_datasets: int = 1) -> dict[str, float]:
    """Estimate expected wall-clock runtime in seconds before starting a benchmark."""
    if isinstance(cfg, dict):
        base = load_default_config()
        merged = {**base, **cfg}
        parsed = BenchmarkConfig.model_validate(
            {k: v for k, v in merged.items() if k in BenchmarkConfig.model_fields}
        )
    else:
        parsed = cfg

    specs = get_models(None if parsed.models == ["all"] else parsed.models) if parsed.models else []
    classic_per_model = 15.0 if parsed.tuning != "none" else 2.0
    classic_total = len(specs) * classic_per_model

    automl_total = 0.0
    for _ in parsed.automl_frameworks:
        automl_total += float(parsed.automl_budget_s + parsed.automl_grace_s)

    single_run = (classic_total + automl_total) * max(1, parsed.outer_folds)
    total_seconds = single_run * n_datasets

    return {
        "classic_seconds": round(classic_total * parsed.outer_folds * n_datasets, 1),
        "automl_seconds": round(automl_total * parsed.outer_folds * n_datasets, 1),
        "total_seconds": round(total_seconds, 1),
        "total_minutes": round(total_seconds / 60.0, 1),
    }


def run_benchmark(
    dataset: LoadedDataset,
    cfg: dict | None = None,
    progress_callback: ProgressCb = None,
    should_stop: Callable[[], bool] | None = None,
) -> BenchmarkResult:
    """Train models sequentially so timings stay comparable; folds may use n_jobs.

    ``should_stop`` is polled between models: when it returns True the run ends
    early and returns the results collected so far (status ``cancelled`` is set by
    the caller, not here).
    """
    base = load_default_config()
    if cfg is not None:
        cfg_dict = cfg.model_dump() if hasattr(cfg, "model_dump") else cfg
        base.update({k: v for k, v in cfg_dict.items() if v is not None})
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
    specs = get_models(None if parsed.models == ["all"] else parsed.models) if parsed.models else []
    classes = list(range(int(pd.Series(dataset.y).nunique())))
    results: list[ModelRunResult] = []
    deadline = time.perf_counter() + float(parsed.total_budget_seconds)
    total_steps = len(specs) + len(parsed.automl_frameworks)

    # Tier 1: Classic ML models (leak-free scikit-learn pipeline)
    for i, spec in enumerate(specs, start=1):
        if should_stop and should_stop():
            logger.info("benchmark stopped before %s; returning partial results", spec.name)
            break
        if progress_callback:
            progress_callback(
                spec.name, i - 1, total_steps, f"Training {spec.name}", status="running"
            )
        if time.perf_counter() > deadline:
            results.append(
                ModelRunResult(
                    model_name=spec.name,
                    kind="classic",
                    status="timeout",
                    error="Total time budget exhausted",
                )
            )
            if progress_callback:
                progress_callback(
                    spec.name, i, total_steps, f"Skipped {spec.name}", status="timeout"
                )
            continue
        t_model = time.perf_counter()
        res = _run_one(spec, X_train, y_train, X_test, y_test, profile, merged, cv, classes)
        res.kind = "classic"
        results.append(res)
        if progress_callback:
            progress_callback(
                spec.name,
                i,
                total_steps,
                f"Finished {spec.name}",
                status=res.status,
                elapsed_s=round(time.perf_counter() - t_model, 3),
            )

    # Tier 2: AutoML contender frameworks (run in isolated environments with raw data)
    for k, fid in enumerate(parsed.automl_frameworks, start=len(specs) + 1):
        if should_stop and should_stop():
            logger.info("benchmark stopped before %s; returning partial results", fid)
            break
        if progress_callback:
            progress_callback(
                fid, k - 1, total_steps, f"Running AutoML framework {fid}", status="running"
            )
        if time.perf_counter() > deadline:
            results.append(
                ModelRunResult(
                    model_name=fid,
                    kind="automl",
                    framework=fid,
                    budget_s=parsed.automl_budget_s,
                    status="timeout",
                    error="Total time budget exhausted",
                )
            )
            if progress_callback:
                progress_callback(fid, k, total_steps, f"Skipped {fid}", status="timeout")
            continue
        t_model = time.perf_counter()
        try:
            train_spec = TrainSpec.from_arrays(
                framework_id=fid,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                task_type="binary" if len(classes) == 2 else "multiclass",
                n_classes=len(classes),
                time_budget_s=parsed.automl_budget_s,
                seed=parsed.seed,
                metric=parsed.metrics_primary,
            )
            run_res = run_framework(train_spec, grace_extra_s=parsed.automl_grace_s)
            eval_res = evaluate_automl_result(run_res, y_test, classes)
            budget_used_ratio = (
                round(run_res.fit_time_s / max(1, parsed.automl_budget_s), 3)
                if run_res.fit_time_s > 0
                else 0.0
            )
            interp = 1 if fid in {"autogluon", "h2o", "lightautoml"} else 2
            results.append(
                ModelRunResult(
                    model_name=fid,
                    kind="automl",
                    framework=fid,
                    framework_version=run_res.framework_version,
                    status=run_res.status,
                    error=run_res.error,
                    notes=run_res.best_pipeline or "",
                    test_metrics=eval_res["test_metrics"],
                    training_time=run_res.fit_time_s,
                    prediction_time_per_1k=run_res.predict_time_s,
                    budget_s=parsed.automl_budget_s,
                    budget_used_ratio=budget_used_ratio,
                    model_summary=run_res.best_pipeline,
                    interpretability_score=interp,
                    y_pred=eval_res["y_pred"],
                    y_true=eval_res["y_true"],
                    y_proba=eval_res["y_proba"],
                )
            )
        except Exception as exc:
            logger.exception("AutoML framework %s failed with unhandled exception", fid)
            results.append(
                ModelRunResult(
                    model_name=fid,
                    kind="automl",
                    framework=fid,
                    budget_s=parsed.automl_budget_s,
                    status="failed",
                    error=str(exc),
                )
            )
        if progress_callback:
            progress_callback(
                fid,
                k,
                total_steps,
                f"Finished AutoML framework {fid}",
                status=results[-1].status,
                elapsed_s=round(time.perf_counter() - t_model, 3),
            )

    return BenchmarkResult(
        profile=profile.model_dump(),
        config=merged,
        seed=parsed.seed,
        n_jobs=parsed.n_jobs,
        lib_versions=library_versions(),
        models=results,
        classes=list(dataset.meta.get("classes", [str(c) for c in classes])),
    )
