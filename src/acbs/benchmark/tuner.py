from __future__ import annotations

from typing import Any

import optuna
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

from acbs.models.registry import ModelSpec, apply_class_weight
from acbs.models.search_spaces import param_distributions, suggest_params
from acbs.preprocessing.pipeline import build_full_pipeline
from acbs.utils.logging import get_logger

logger = get_logger(__name__)


def _make_estimator(spec: ModelSpec, params: dict, seed: int, n_jobs: int, y, strategy: str):
    est = spec.estimator_factory(params, seed, n_jobs)
    apply_class_weight(est, strategy, y)
    return est


def tune_model(
    spec: ModelSpec,
    X_train,
    y_train,
    preprocessor,
    cfg,
    cv,
    imbalance_strategy: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    method = cfg.get("tuning", "none")
    seed = int(cfg.get("seed", 42))
    n_jobs = int(cfg.get("n_jobs", 1))
    primary = cfg.get("metrics_primary", "f1_macro")
    n_trials = int(cfg.get("n_trials", 20))
    timeout = float(cfg.get("time_budget_seconds", 120))

    if method in {None, "none"}:
        return {}, []

    if method == "random":
        est = _make_estimator(spec, {}, seed, n_jobs, y_train, imbalance_strategy)
        pipe = build_full_pipeline(est, preprocessor, imbalance_strategy)
        search = RandomizedSearchCV(
            pipe,
            param_distributions={f"clf__{k}": v for k, v in param_distributions(spec.name).items()},
            n_iter=min(n_trials, 15),
            cv=cv,
            scoring=primary if primary != "f1_macro" else "f1_macro",
            random_state=seed,
            n_jobs=n_jobs,
        )
        search.fit(X_train, y_train)
        best = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        return best, [{"score": float(search.best_score_)}]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, spec.name)
        est = _make_estimator(spec, params, seed, n_jobs, y_train, imbalance_strategy)
        pipe = build_full_pipeline(est, preprocessor, imbalance_strategy)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=primary, n_jobs=n_jobs)
        return float(scores.mean())

    study.optimize(objective, n_trials=n_trials, timeout=timeout, catch=(Exception,))
    history = [{"number": t.number, "value": t.value, "params": t.params} for t in study.trials]
    # ``study.best_trial`` raises ValueError when every trial failed, so it cannot
    # be used as a truthiness guard. Fall back to default params instead.
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        logger.warning(
            "all %d tuning trials failed for %s; using default parameters",
            len(study.trials),
            spec.name,
        )
        return {}, history
    return dict(study.best_params), history
