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


def test_progress_callback_reports_status_and_elapsed():
    np.random.seed(42)
    n = 80
    X = pd.DataFrame(np.random.randn(n, 4), columns=[f"f{i}" for i in range(4)])
    y = pd.Series(np.random.choice([0, 1], size=n))
    dataset = LoadedDataset(X=X, y=y, meta={"target_column": "target", "classes": ["0", "1"]})

    events = []

    def cb(name, step, total, message, status=None, elapsed_s=None):
        events.append((name, status, elapsed_s, step, total))

    cfg = {"models": ["logistic_regression", "decision_tree"], "cv_folds": 2, "tuning": "none"}
    res = run_benchmark(dataset, cfg, progress_callback=cb)

    assert len(res.models) == 2
    names = [e[0] for e in events]
    assert names[0] == "logistic_regression"
    # every model reports a running event and a terminal event with timing
    for model in ("logistic_regression", "decision_tree"):
        statuses = [e[1] for e in events if e[0] == model]
        assert statuses[0] == "running"
        assert statuses[-1] in {"ok", "failed", "timeout", "skipped"}
        finished = [e for e in events if e[0] == model and e[1] == "ok"]
        assert finished and finished[-1][2] is not None


def test_should_stop_returns_partial_results():
    np.random.seed(42)
    n = 80
    X = pd.DataFrame(np.random.randn(n, 4), columns=[f"f{i}" for i in range(4)])
    y = pd.Series(np.random.choice([0, 1], size=n))
    dataset = LoadedDataset(X=X, y=y, meta={"target_column": "target", "classes": ["0", "1"]})

    calls = {"n": 0}

    def should_stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 1  # allow the first model, stop before the second

    cfg = {
        "models": ["logistic_regression", "decision_tree", "naive_bayes"],
        "cv_folds": 2,
        "tuning": "none",
    }
    res = run_benchmark(dataset, cfg, should_stop=should_stop)

    assert len(res.models) == 1
    assert res.models[0].model_name == "logistic_regression"
    assert res.models[0].status == "ok"


def test_statistical_comparison():
    from acbs.benchmark.aggregate import compare_models_statistically

    fold_scores = pd.DataFrame(
        {
            "model_a": [0.8, 0.82, 0.81, 0.83, 0.80],
            "model_b": [0.75, 0.74, 0.76, 0.73, 0.75],
            "model_c": [0.81, 0.81, 0.82, 0.82, 0.81],
        }
    )

    res = compare_models_statistically(fold_scores)
    assert "friedman_p" in res
    assert "best" in res


def test_estimate_cost():
    from acbs.benchmark.engine import estimate_cost

    cfg = {
        "models": ["logistic_regression", "decision_tree"],
        "automl_frameworks": ["flaml", "tpot"],
        "automl_budget_s": 60,
        "automl_grace_s": 30,
        "tuning": "none",
        "outer_folds": 1,
    }
    cost = estimate_cost(cfg, n_datasets=1)
    assert cost["classic_seconds"] == 4.0  # 2 models * 2s
    assert cost["automl_seconds"] == 180.0  # 2 frameworks * (60 + 30)
    assert cost["total_seconds"] == 184.0
    assert cost["total_minutes"] > 0


def test_run_benchmark_with_automl():
    from unittest.mock import patch

    from acbs.automl.contracts import RunResult

    np.random.seed(42)
    n = 60
    X = pd.DataFrame(np.random.randn(n, 4), columns=[f"f{i}" for i in range(4)])
    y = pd.Series(np.random.choice([0, 1], size=n))
    dataset = LoadedDataset(X=X, y=y, meta={"target_column": "target", "classes": ["0", "1"]})

    cfg = {
        "models": ["dummy"],
        "automl_frameworks": ["fake_ok", "fake_fail"],
        "automl_budget_s": 30,
        "tuning": "none",
    }

    def fake_run(spec, grace_extra_s=60):
        if spec.framework_id == "fake_ok":
            n_test = len(spec.X_test)
            return RunResult(
                framework_id=spec.framework_id,
                status="ok",
                proba=[[0.3, 0.7] for _ in range(n_test)],
                fit_time_s=1.5,
                predict_time_s=0.02,
                framework_version="1.0",
                best_pipeline="FakeEnsemble",
            )
        return RunResult.failed(spec.framework_id, "Mocked crash in isolated worker")

    with patch("acbs.benchmark.engine.run_framework", side_effect=fake_run):
        res = run_benchmark(dataset, cfg)

    assert len(res.models) == 3
    # Check dummy (classic)
    dummy_res = next(m for m in res.models if m.model_name == "dummy")
    assert dummy_res.kind == "classic"
    assert dummy_res.status == "ok"

    # Check fake_ok (automl)
    ok_res = next(m for m in res.models if m.model_name == "fake_ok")
    assert ok_res.kind == "automl"
    assert ok_res.status == "ok"
    assert "accuracy" in ok_res.test_metrics
    assert ok_res.training_time == 1.5

    # Check fake_fail (automl)
    fail_res = next(m for m in res.models if m.model_name == "fake_fail")
    assert fail_res.kind == "automl"
    assert fail_res.status == "failed"
    assert "Mocked crash" in (fail_res.error or "")
