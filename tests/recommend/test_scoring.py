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
    return pd.DataFrame(
        [
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
        ]
    )


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
    exp = generate_explanation(
        winner_row, sample_results_df, {"is_imbalanced": False}, pref, stability
    )

    assert "headline" in exp
    assert exp["model_name"] == winner_row["model_name"]
    assert len(exp["paragraphs"]) > 0


def test_automl_value_analysis():
    from acbs.recommend.scoring import automl_value_analysis

    # Case 1: AutoML wins by 5 points (+0.05 F1) at 4x time -> "AutoML worth it"
    df_worth_it = pd.DataFrame(
        [
            {
                "model_name": "random_forest",
                "kind": "classic",
                "status": "ok",
                "f1_macro": 0.85,
                "accuracy": 0.86,
                "training_time": 2.0,
                "prediction_time_per_1k": 0.05,
            },
            {
                "model_name": "autogluon",
                "kind": "automl",
                "status": "ok",
                "f1_macro": 0.90,  # +5 points
                "accuracy": 0.91,
                "training_time": 8.0,  # 4x time
                "prediction_time_per_1k": 0.10,
            },
        ]
    )
    res1 = automl_value_analysis(df_worth_it)
    assert res1["has_both_tiers"] is True
    assert res1["verdict"] == "AutoML worth it"
    assert res1["delta_f1"] == 0.05
    assert res1["best_classic"] == "random_forest"
    assert res1["best_automl"] == "autogluon"

    # Case 2: AutoML wins by only 0.5 points (+0.005 F1) at 50x time -> "Not worth it"
    df_not_worth = pd.DataFrame(
        [
            {
                "model_name": "random_forest",
                "kind": "classic",
                "status": "ok",
                "f1_macro": 0.890,
                "accuracy": 0.890,
                "training_time": 1.0,
                "prediction_time_per_1k": 0.02,
            },
            {
                "model_name": "h2o",
                "kind": "automl",
                "status": "ok",
                "f1_macro": 0.895,  # +0.005 points
                "accuracy": 0.895,
                "training_time": 50.0,  # 50x time
                "prediction_time_per_1k": 0.20,
            },
        ]
    )
    res2 = automl_value_analysis(df_not_worth)
    assert res2["has_both_tiers"] is True
    # With tie_f1_delta = 0.008, 0.005 is within tie threshold, delta_f1 < 0.02 so it's Marginal or Not worth it
    assert res2["verdict"] in {"Not worth it", "Marginal"}
    assert res2["is_statistical_tie"] is True

    # Case 3: Classic model outright beats AutoML -> "Not worth it"
    df_classic_wins = pd.DataFrame(
        [
            {
                "model_name": "xgboost",
                "kind": "classic",
                "status": "ok",
                "f1_macro": 0.92,
                "accuracy": 0.92,
                "training_time": 2.0,
                "prediction_time_per_1k": 0.03,
            },
            {
                "model_name": "flaml",
                "kind": "automl",
                "status": "ok",
                "f1_macro": 0.88,
                "accuracy": 0.88,
                "training_time": 20.0,
                "prediction_time_per_1k": 0.05,
            },
        ]
    )
    res3 = automl_value_analysis(df_classic_wins)
    assert res3["verdict"] == "Not worth it"


def test_weights_favoring_speed_pick_classic():
    df = pd.DataFrame(
        [
            {
                "model_name": "logistic_regression",
                "kind": "classic",
                "status": "ok",
                "accuracy": 0.85,
                "f1_macro": 0.84,
                "precision_macro": 0.85,
                "recall_macro": 0.83,
                "roc_auc": 0.88,
                "training_time": 0.1,
                "prediction_time_per_1k": 0.001,
                "interpretability_score": 5,
            },
            {
                "model_name": "autogluon",
                "kind": "automl",
                "framework": "autogluon",
                "status": "ok",
                "accuracy": 0.88,
                "f1_macro": 0.87,
                "precision_macro": 0.88,
                "recall_macro": 0.86,
                "roc_auc": 0.91,
                "training_time": 60.0,
                "prediction_time_per_1k": 0.150,  # 150x slower predict
                "interpretability_score": 1,
            },
        ]
    )
    pref = UserPreferences(preset="fast_inference")
    scored = score_models(df, pref)
    winner = scored.sort_values("rank").iloc[0]["model_name"]
    assert winner == "logistic_regression"

    # And verify explanation generation
    exp = generate_explanation(scored.iloc[0], df, {}, pref)
    assert "logistic_regression" in exp["model_name"]
    assert "AutoML Comparison" in " ".join(exp["paragraphs"])
