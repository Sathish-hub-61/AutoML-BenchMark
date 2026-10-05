from __future__ import annotations

import numpy as np
import pandas as pd

from acbs.benchmark.engine import run_benchmark
from acbs.data.loader import LoadedDataset


def test_run_benchmark_synthetic():
    np.random.seed(42)
    n = 100
    X = pd.DataFrame(np.random.randn(n, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(np.random.choice([0, 1], size=n))
    dataset = LoadedDataset(X=X, y=y, meta={"target_column": "target", "classes": ["0", "1"]})

    cfg = {
        "models": ["logistic_regression", "decision_tree", "naive_bayes"],
        "cv_folds": 3,
        "seed": 42,
        "tuning": "none",
        "time_budget_seconds": 30,
    }

    res = run_benchmark(dataset, cfg)

    assert res.seed == 42
    assert len(res.models) == 3
    for model_res in res.models:
        assert model_res.status in {"ok", "skipped", "failed", "timeout"}
        if model_res.status == "ok":
            assert "accuracy" in model_res.test_metrics
            assert "f1_macro" in model_res.test_metrics
            assert model_res.training_time is not None
            assert model_res.prediction_time_per_1k is not None


def test_statistical_comparison():
    from acbs.benchmark.aggregate import compare_models_statistically

    fold_scores = pd.DataFrame({
        "model_a": [0.8, 0.82, 0.81, 0.83, 0.80],
        "model_b": [0.75, 0.74, 0.76, 0.73, 0.75],
        "model_c": [0.81, 0.81, 0.82, 0.82, 0.81],
    })

    res = compare_models_statistically(fold_scores)
    assert "friedman_p" in res
    assert "best" in res
