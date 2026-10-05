from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

EstimatorFactory = Callable[[dict[str, Any], int, int], Any]


@dataclass
class ModelSpec:
    name: str
    family: str
    estimator_factory: EstimatorFactory
    needs_scaling: bool
    supports_class_weight: bool
    supports_proba: bool
    interpretability_score: int
    scalability_note: str
    max_rows: int | None = None


def _lr(params: dict, seed: int, n_jobs: int) -> LogisticRegression:
    merged = {"max_iter": 400, "solver": "lbfgs", **params}
    return LogisticRegression(random_state=seed, n_jobs=n_jobs if "n_jobs" in LogisticRegression().get_params() else None, **{k: v for k, v in merged.items() if k != "n_jobs"})


def _svm(params: dict, seed: int, n_jobs: int, n_rows: int | None = None) -> Any:
    max_rows = 20_000
    if n_rows is not None and n_rows > max_rows:
        base = LinearSVC(random_state=seed, dual="auto", **{k: v for k, v in params.items() if k in {"C"}})
        return CalibratedClassifierCV(base, cv=3)
    merged = {"probability": True, "kernel": "rbf", **params}
    return SVC(random_state=seed, **merged)


def _dt(params: dict, seed: int, n_jobs: int) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(random_state=seed, **params)


def _rf(params: dict, seed: int, n_jobs: int) -> RandomForestClassifier:
    merged = {"n_estimators": 150, **params}
    return RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **merged)


def _xgb(params: dict, seed: int, n_jobs: int) -> XGBClassifier:
    merged = {
        "n_estimators": 150,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "verbosity": 0,
        **params,
    }
    return XGBClassifier(random_state=seed, n_jobs=n_jobs, **merged)


def _gb(params: dict, seed: int, n_jobs: int) -> GradientBoostingClassifier:
    merged = {"n_estimators": 100, **params}
    return GradientBoostingClassifier(random_state=seed, **merged)


def _nb(params: dict, seed: int, n_jobs: int) -> GaussianNB:
    return GaussianNB(**params)


def _knn(params: dict, seed: int, n_jobs: int) -> KNeighborsClassifier:
    merged = {"n_neighbors": 5, **params}
    return KNeighborsClassifier(n_jobs=n_jobs, **merged)


_REGISTRY: dict[str, ModelSpec] = {}


def register_model(spec: ModelSpec) -> ModelSpec:
    _REGISTRY[spec.name] = spec
    return spec


def _bootstrap() -> None:
    if _REGISTRY:
        return
    register_model(
        ModelSpec(
            "logistic_regression",
            "linear",
            _lr,
            True,
            True,
            True,
            5,
            "Fast linear baseline",
        )
    )
    register_model(
        ModelSpec("svm", "svm", lambda p, s, n: _svm(p, s, n), True, True, True, 2, "Slow on large n", 20_000)
    )
    register_model(ModelSpec("decision_tree", "tree", _dt, False, True, True, 5, "Interpretable"))
    register_model(ModelSpec("random_forest", "ensemble", _rf, False, True, True, 3, "Strong tabular default"))
    register_model(ModelSpec("xgboost", "boosting", _xgb, False, False, True, 2, "Often best accuracy"))
    register_model(ModelSpec("gradient_boosting", "boosting", _gb, False, False, True, 2, "sklearn boosting"))
    register_model(ModelSpec("naive_bayes", "naive_bayes", _nb, True, False, True, 4, "Very fast"))
    register_model(ModelSpec("knn", "instance", _knn, True, False, True, 3, "Slow predict on large n", 100_000))


def get_models(names: list[str] | None = None) -> list[ModelSpec]:
    _bootstrap()
    if not names or names == ["all"]:
        return list(_REGISTRY.values())
    missing = [n for n in names if n not in _REGISTRY]
    if missing:
        raise KeyError(f"Unknown models: {missing}. Available: {list(_REGISTRY)}")
    return [_REGISTRY[n] for n in names]


def apply_class_weight(estimator, strategy: str, y) -> None:
    if strategy != "class_weight":
        return
    params = estimator.get_params(deep=False)
    if "class_weight" in params:
        estimator.set_params(class_weight="balanced")
    if estimator.__class__.__name__ == "XGBClassifier":
        import numpy as np

        values, counts = np.unique(y, return_counts=True)
        if len(values) == 2:
            majority = counts.max()
            minority = max(counts.min(), 1)
            estimator.set_params(scale_pos_weight=float(majority / minority))
