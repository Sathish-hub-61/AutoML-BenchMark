from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from acbs.benchmark.tuner import tune_model
from acbs.data.profiler import profile_dataset
from acbs.data.splitter import make_cv_splitter
from acbs.models.registry import get_models
from acbs.preprocessing.pipeline import build_preprocessor


def _tiny_dataset():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60)})
    y = pd.Series(rng.choice([0, 1], size=60))
    return X, y


def test_all_tuning_trials_failed_returns_default_params():
    """study.best_trial raises when no trial completed; tuner must degrade, not crash."""
    X, y = _tiny_dataset()
    profile = profile_dataset(X, y)
    cfg = {"tuning": "optuna", "n_trials": 3, "seed": 42, "metrics_primary": "f1_macro"}
    spec = get_models(["logistic_regression"])[0]
    preprocessor = build_preprocessor(profile, cfg, spec.family, X)
    cv = make_cv_splitter(cv_folds=3, seed=42)

    with patch(
        "acbs.benchmark.tuner.cross_val_score",
        side_effect=RuntimeError("every trial blows up"),
    ):
        best, history = tune_model(spec, X, y, preprocessor, cfg, cv, "none")

    assert best == {}
    assert len(history) == 3
    assert all(h["value"] is None for h in history)
