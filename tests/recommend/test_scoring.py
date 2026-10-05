from __future__ import annotations

import pandas as pd
import pytest

from acbs.recommend.explain import generate_explanation
from acbs.recommend.scoring import (
    UserPreferences,
    pareto_front,
    score_models,
    sensitivity_analysis,
)


@pytest.fixture
def sample_results_df():
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
        {
            "model_name": "xgboost",
            "status": "ok",
            "accuracy": 0.92,
            "f1_macro": 0.91,
            "precision_macro": 0.92,
            "recall_macro": 0.90,
            "roc_auc": 0.96,
            "training_time": 4.0,
            "prediction_time_per_1k": 0.08,
            "interpretability_score": 2,
        },
    ])


def test_user_preferences_presets():
    pref = UserPreferences(preset="fast_inference")
    assert pref.predict_time > pref.accuracy
    weights = pref.metric_weights()
    assert sum(weights.values()) == pytest.approx(1.0)


def test_score_models_and_pareto(sample_results_df):
    pref_acc = UserPreferences(preset="max_accuracy")
    scored = score_models(sample_results_df, pref_acc)
    top_model = scored.sort_values("rank").iloc[0]["model_name"]
    assert top_model == "xgboost"

    pref_fast = UserPreferences(preset="fast_inference")
    scored_fast = score_models(sample_results_df, pref_fast)
    top_fast = scored_fast.sort_values("rank").iloc[0]["model_name"]
    assert top_fast == "logistic_regression"

    p_front = pareto_front(sample_results_df, x="training_time", y="f1_macro")
    assert "logistic_regression" in p_front
    assert "xgboost" in p_front


def test_constraints_filter(sample_results_df):
    pref = UserPreferences(max_training_time_s=1.0)
    scored = score_models(sample_results_df, pref)
    excluded = scored[scored["excluded"]]["model_name"].tolist()
    assert "random_forest" in excluded
    assert "xgboost" in excluded


def test_sensitivity_analysis_and_explanation(sample_results_df):
    pref = UserPreferences(preset="balanced")
    stability = sensitivity_analysis(sample_results_df, pref)
    assert isinstance(stability, dict)

    scored = score_models(sample_results_df, pref)
    winner_row = scored.iloc[0]
    exp = generate_explanation(winner_row, sample_results_df, {"is_imbalanced": False}, pref, stability)

    assert "headline" in exp
    assert exp["model_name"] == winner_row["model_name"]
    assert len(exp["paragraphs"]) > 0
