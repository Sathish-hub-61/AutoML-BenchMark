from __future__ import annotations

import pandas as pd
import pytest

from acbs.benchmark.engine import BenchmarkResult
from acbs.recommend.scoring import UserPreferences
from acbs.reporting.charts import (
    accuracy_vs_time_scatter,
    grouped_metrics_bar,
    metric_radar,
    performance_bar,
    time_comparison,
)
from acbs.reporting.report import build_html_report, export_csv


@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {
            "model_name": "logistic_regression",
            "status": "ok",
            "accuracy": 0.85,
            "f1_macro": 0.84,
            "precision_macro": 0.85,
            "recall_macro": 0.83,
            "roc_auc": 0.90,
            "training_time": 0.5,
            "prediction_time_per_1k": 0.01,
            "interpretability_score": 5,
        },
        {
            "model_name": "random_forest",
            "status": "ok",
            "accuracy": 0.90,
            "f1_macro": 0.89,
            "precision_macro": 0.90,
            "recall_macro": 0.88,
            "roc_auc": 0.95,
            "training_time": 2.5,
            "prediction_time_per_1k": 0.05,
            "interpretability_score": 3,
        },
    ])


def test_chart_generation(sample_df):
    fig1 = performance_bar(sample_df, "accuracy")
    assert fig1 is not None
    fig2 = grouped_metrics_bar(sample_df)
    assert fig2 is not None
    fig3 = metric_radar(sample_df, ["logistic_regression", "random_forest"], ["accuracy", "f1_macro"])
    assert fig3 is not None
    fig4 = time_comparison(sample_df)
    assert fig4 is not None
    fig5 = accuracy_vs_time_scatter(sample_df)
    assert fig5 is not None


def test_html_report_building(sample_df):
    res_dict = {
        "profile": {"n_rows": 100, "n_features": 5, "task_type": "binary"},
        "config": {"models": ["logistic_regression", "random_forest"]},
        "seed": 42,
        "n_jobs": 1,
        "lib_versions": {"scikit-learn": "1.3.0"},
        "models": [
            {
                "model_name": "logistic_regression",
                "status": "ok",
                "test_metrics": {"accuracy": 0.85, "f1_macro": 0.84},
                "training_time": 0.5,
                "prediction_time_per_1k": 0.01,
            }
        ],
        "classes": ["0", "1"],
    }
    bresult = BenchmarkResult.model_validate(res_dict)
    prefs = UserPreferences(preset="balanced")

    html = build_html_report(bresult, prefs)
    assert "<html" in html.lower()
    assert "logistic_regression" in html

    csv_data = export_csv(sample_df)
    assert isinstance(csv_data, str)
