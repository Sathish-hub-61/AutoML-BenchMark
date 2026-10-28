from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, model_validator

BENEFIT = [
    "accuracy",
    "f1_macro",
    "precision_macro",
    "recall_macro",
    "roc_auc",
    "interpretability_score",
]
COST = ["training_time", "prediction_time_per_1k", "budget_used_ratio"]

PRESETS = {
    "balanced": {
        "accuracy": 0.15,
        "f1": 0.25,
        "precision": 0.1,
        "recall": 0.15,
        "roc_auc": 0.15,
        "train_time": 0.1,
        "predict_time": 0.05,
        "interpretability": 0.05,
        "budget_used": 0.0,
    },
    "max_accuracy": {
        "accuracy": 0.35,
        "f1": 0.3,
        "precision": 0.1,
        "recall": 0.1,
        "roc_auc": 0.15,
        "train_time": 0.0,
        "predict_time": 0.0,
        "interpretability": 0.0,
        "budget_used": 0.0,
    },
    "fast_inference": {
        "accuracy": 0.1,
        "f1": 0.15,
        "precision": 0.05,
        "recall": 0.05,
        "roc_auc": 0.05,
        "train_time": 0.15,
        "predict_time": 0.35,
        "interpretability": 0.10,
        "budget_used": 0.0,
    },
    "interpretable": {
        "accuracy": 0.15,
        "f1": 0.15,
        "precision": 0.1,
        "recall": 0.1,
        "roc_auc": 0.1,
        "train_time": 0.05,
        "predict_time": 0.05,
        "interpretability": 0.3,
        "budget_used": 0.0,
    },
    "imbalance_safe": {
        "accuracy": 0.05,
        "f1": 0.35,
        "precision": 0.1,
        "recall": 0.35,
        "roc_auc": 0.1,
        "train_time": 0.025,
        "predict_time": 0.025,
        "interpretability": 0.0,
        "budget_used": 0.0,
    },
}


class UserPreferences(BaseModel):
    accuracy: float = 0.15
    f1: float = 0.25
    precision: float = 0.1
    recall: float = 0.15
    roc_auc: float = 0.15
    train_time: float = 0.1
    predict_time: float = 0.05
    interpretability: float = 0.05
    budget_used: float = 0.0
    max_training_time_s: float | None = None
    max_prediction_time_ms_per_1k: float | None = None
    min_recall: float | None = None
    min_precision: float | None = None
    method: Literal["weighted_sum", "topsis"] = "weighted_sum"
    preset: str | None = None

    @model_validator(mode="after")
    def _apply_preset_and_norm(self):
        if self.preset and self.preset in PRESETS:
            for k, v in PRESETS[self.preset].items():
                setattr(self, k, v)
        weights = np.array(
            [
                self.accuracy,
                self.f1,
                self.precision,
                self.recall,
                self.roc_auc,
                self.train_time,
                self.predict_time,
                self.interpretability,
                self.budget_used,
            ],
            dtype=float,
        )
        weights = np.clip(weights, 0, None)
        total = weights.sum()
        if total == 0:
            weights[:] = 1 / len(weights)
        else:
            weights = weights / total
        (
            self.accuracy,
            self.f1,
            self.precision,
            self.recall,
            self.roc_auc,
            self.train_time,
            self.predict_time,
            self.interpretability,
            self.budget_used,
        ) = [float(x) for x in weights]
        return self

    def metric_weights(self) -> dict[str, float]:
        w = {
            "accuracy": self.accuracy,
            "f1_macro": self.f1,
            "precision_macro": self.precision,
            "recall_macro": self.recall,
            "roc_auc": self.roc_auc,
            "training_time": self.train_time,
            "prediction_time_per_1k": self.predict_time,
            "interpretability_score": self.interpretability,
        }
        if self.budget_used > 0:
            w["budget_used_ratio"] = self.budget_used
        return w


def _minmax(series: pd.Series, invert: bool = False) -> pd.Series:
    s = series.astype(float)
    if invert:
        s = np.log1p(s.clip(lower=0))
        s = -s
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(np.ones(len(s)) * 0.5, index=s.index)
    return (s - lo) / (hi - lo)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    # Prepare fallbacks for training time and prediction time if needed
    train_time_series = (
        df["training_time"] if "training_time" in df.columns else pd.Series(np.nan, index=df.index)
    )
    if "fit_time_s" in df.columns:
        train_time_series = train_time_series.fillna(df["fit_time_s"])

    pred_time_series = (
        df["prediction_time_per_1k"]
        if "prediction_time_per_1k" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    if "predict_time_s" in df.columns:
        pred_time_series = pred_time_series.fillna(df["predict_time_s"])

    mapping_present = [c for c in BENEFIT if c in df.columns]
    for col in mapping_present:
        out[col] = _minmax(
            df[col].fillna(df[col].median() if df[col].notna().any() else 0), invert=False
        )

    if "training_time" in df.columns or "fit_time_s" in df.columns:
        s_train = pd.to_numeric(train_time_series, errors="coerce")
        out["training_time"] = _minmax(
            s_train.fillna(float(s_train.max()) if s_train.notna().any() else 1.0), invert=True
        )

    if "prediction_time_per_1k" in df.columns or "predict_time_s" in df.columns:
        s_pred = pd.to_numeric(pred_time_series, errors="coerce")
        out["prediction_time_per_1k"] = _minmax(
            s_pred.fillna(float(s_pred.max()) if s_pred.notna().any() else 1.0), invert=True
        )

    if "budget_used_ratio" in df.columns:
        s_budget = pd.to_numeric(df["budget_used_ratio"], errors="coerce")
        out["budget_used_ratio"] = _minmax(s_budget.fillna(1.0), invert=True)

    return out


def _apply_constraints(df: pd.DataFrame, prefs: UserPreferences) -> pd.DataFrame:
    reasons = []
    keep = []
    for _, row in df.iterrows():
        why = []
        if row.get("status") != "ok":
            why.append(f"status={row.get('status')}")
        if prefs.max_training_time_s is not None and pd.notna(row.get("training_time")):
            if row["training_time"] > prefs.max_training_time_s:
                why.append("max_training_time")
        if prefs.max_prediction_time_ms_per_1k is not None and pd.notna(
            row.get("prediction_time_per_1k")
        ):
            if row["prediction_time_per_1k"] * 1000 > prefs.max_prediction_time_ms_per_1k:
                why.append("max_prediction_time")
        if prefs.min_recall is not None and pd.notna(row.get("recall_macro")):
            if row["recall_macro"] < prefs.min_recall:
                why.append("min_recall")
        if prefs.min_precision is not None and pd.notna(row.get("precision_macro")):
            if row["precision_macro"] < prefs.min_precision:
                why.append("min_precision")
        reasons.append("; ".join(why) if why else "")
        keep.append(len(why) == 0)
    df = df.copy()
    df["excluded"] = [not k for k in keep]
    df["exclude_reason"] = reasons
    return df


def _weighted_sum(norm: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    score = pd.Series(0.0, index=norm.index)
    for col, w in weights.items():
        if col in norm.columns and w:
            score = score + w * norm[col]
    return score


def _topsis(norm: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    cols = [c for c in weights if c in norm.columns and weights[c] > 0]
    if not cols:
        return pd.Series(0.0, index=norm.index)
    mat = norm[cols].to_numpy(dtype=float)
    w = np.array([weights[c] for c in cols], dtype=float)
    weighted = mat * w
    ideal = weighted.max(axis=0)
    nadir = weighted.min(axis=0)
    d_plus = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
    d_minus = np.sqrt(((weighted - nadir) ** 2).sum(axis=1))
    denom = d_plus + d_minus
    denom[denom == 0] = 1
    return pd.Series(d_minus / denom, index=norm.index)


def score_models(results_df: pd.DataFrame, prefs: UserPreferences) -> pd.DataFrame:
    df = _apply_constraints(results_df, prefs)
    eligible = df.loc[~df["excluded"]].copy()
    df["composite_score"] = np.nan
    df["rank"] = np.nan
    if eligible.empty:
        return df
    norm = normalize(eligible)
    weights = prefs.metric_weights()
    scores = _topsis(norm, weights) if prefs.method == "topsis" else _weighted_sum(norm, weights)
    eligible = eligible.copy()
    eligible["composite_score"] = scores
    eligible["rank"] = scores.rank(ascending=False, method="min")
    df.loc[eligible.index, "composite_score"] = eligible["composite_score"]
    df.loc[eligible.index, "rank"] = eligible["rank"]
    return df.sort_values(["excluded", "rank", "model_name"], na_position="last")


def pareto_front(
    results_df: pd.DataFrame, x: str = "training_time", y: str = "f1_macro"
) -> list[str]:
    ok = results_df[results_df.get("status", "ok") == "ok"]
    names = []
    for i, row in ok.iterrows():
        dominated = False
        for j, other in ok.iterrows():
            if i == j:
                continue
            better_or_eq_time = other[x] <= row[x]
            better_or_eq_y = other[y] >= row[y]
            strictly = other[x] < row[x] or other[y] > row[y]
            if better_or_eq_time and better_or_eq_y and strictly:
                dominated = True
                break
        if not dominated:
            names.append(row["model_name"])
    return names


def sensitivity_analysis(
    results_df: pd.DataFrame, prefs: UserPreferences, *, n: int = 40
) -> dict[str, float]:
    rng = np.random.default_rng(42)
    counts: dict[str, int] = {}
    fields = [
        "accuracy",
        "f1",
        "precision",
        "recall",
        "roc_auc",
        "train_time",
        "predict_time",
        "interpretability",
    ]
    for _ in range(n):
        noise = {f: getattr(prefs, f) * rng.uniform(0.8, 1.2) for f in fields}
        p = prefs.model_copy(update={**noise, "preset": None})
        scored = score_models(results_df, p)
        eligible = scored[~scored["excluded"]]
        if eligible.empty:
            continue
        winner = eligible.sort_values("rank").iloc[0]["model_name"]
        counts[winner] = counts.get(winner, 0) + 1
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def automl_value_analysis(
    results_df: pd.DataFrame,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compare the top AutoML framework against the top classic ML model.

    Determines if AutoML delivers enough predictive improvement to justify
    the increased training time and inference overhead.
    """
    default_thresh = {
        "f1_worth_it_delta": 0.02,
        "max_training_time_ratio": 20.0,
        "max_inference_time_ratio": 10.0,
        "tie_f1_delta": 0.008,
    }
    if thresholds:
        default_thresh.update(thresholds)

    f1_delta_threshold = default_thresh["f1_worth_it_delta"]
    time_ratio_threshold = default_thresh["max_training_time_ratio"]
    tie_threshold = default_thresh["tie_f1_delta"]

    ok_df = results_df[results_df.get("status", "ok") == "ok"].copy()
    if ok_df.empty:
        return {
            "has_both_tiers": False,
            "best_classic": None,
            "best_automl": None,
            "delta_f1": 0.0,
            "delta_roc_auc": 0.0,
            "extra_training_time_ratio": 1.0,
            "extra_inference_time_ratio": 1.0,
            "verdict": "Insufficient data",
            "is_statistical_tie": False,
            "summary": "No models succeeded with status 'ok'.",
        }

    # Identify classic vs automl
    is_automl = ok_df.get("kind", pd.Series("classic", index=ok_df.index)).eq("automl")
    if "framework" in ok_df.columns:
        is_automl = is_automl | (ok_df["framework"].notna() & ok_df["framework"].ne(""))

    classic_df = ok_df[~is_automl]
    automl_df = ok_df[is_automl]

    if classic_df.empty or automl_df.empty:
        best_overall = ok_df.sort_values(by=["f1_macro", "accuracy"], ascending=False).iloc[0][
            "model_name"
        ]
        return {
            "has_both_tiers": False,
            "best_classic": classic_df.iloc[0]["model_name"] if not classic_df.empty else None,
            "best_automl": automl_df.iloc[0]["model_name"] if not automl_df.empty else None,
            "delta_f1": 0.0,
            "delta_roc_auc": 0.0,
            "extra_training_time_ratio": 1.0,
            "extra_inference_time_ratio": 1.0,
            "verdict": "Single tier only",
            "is_statistical_tie": False,
            "summary": f"Benchmark ran only one tier. Top model: {best_overall}.",
        }

    # Best classic model (excluding 'dummy' if other models exist)
    non_dummy_classic = classic_df[classic_df["model_name"] != "dummy"]
    if not non_dummy_classic.empty:
        best_classic_row = non_dummy_classic.sort_values(
            by=["f1_macro", "accuracy"], ascending=False
        ).iloc[0]
    else:
        best_classic_row = classic_df.sort_values(
            by=["f1_macro", "accuracy"], ascending=False
        ).iloc[0]

    # Best AutoML framework
    best_automl_row = automl_df.sort_values(by=["f1_macro", "accuracy"], ascending=False).iloc[0]

    best_classic_name = str(best_classic_row["model_name"])
    best_automl_name = str(best_automl_row["model_name"])

    classic_f1 = float(best_classic_row.get("f1_macro", 0.0))
    automl_f1 = float(best_automl_row.get("f1_macro", 0.0))
    delta_f1 = round(automl_f1 - classic_f1, 4)

    classic_auc = (
        float(best_classic_row.get("roc_auc", 0.0))
        if pd.notna(best_classic_row.get("roc_auc"))
        else 0.0
    )
    automl_auc = (
        float(best_automl_row.get("roc_auc", 0.0))
        if pd.notna(best_automl_row.get("roc_auc"))
        else 0.0
    )
    delta_roc_auc = round(automl_auc - classic_auc, 4)

    classic_ttrain = max(float(best_classic_row.get("training_time", 0.01)), 0.01)
    automl_ttrain = max(float(best_automl_row.get("training_time", 0.01)), 0.01)
    extra_training_time_ratio = round(automl_ttrain / classic_ttrain, 2)

    classic_tpred = max(float(best_classic_row.get("prediction_time_per_1k", 0.001)), 0.0001)
    automl_tpred = max(float(best_automl_row.get("prediction_time_per_1k", 0.001)), 0.0001)
    extra_inference_time_ratio = round(automl_tpred / classic_tpred, 2)

    is_tie = abs(delta_f1) <= tie_threshold

    # Verdict determination
    if delta_f1 >= f1_delta_threshold and extra_training_time_ratio <= time_ratio_threshold:
        verdict = "AutoML worth it"
        summary = (
            f"AutoML is worth it: {best_automl_name} achieves +{delta_f1:.3f} higher F1-macro over {best_classic_name} "
            f"at a reasonable {extra_training_time_ratio:.1f}x training time."
        )
    elif delta_f1 > 0:
        verdict = "Marginal"
        summary = (
            f"AutoML gain is marginal: {best_automl_name} improves F1-macro by only +{delta_f1:.3f} "
            f"while requiring {extra_training_time_ratio:.1f}x training time and {extra_inference_time_ratio:.1f}x inference latency."
        )
    else:
        verdict = "Not worth it"
        summary = (
            f"AutoML is not worth it: classic model {best_classic_name} matched or outperformed {best_automl_name} "
            f"(F1 delta: {delta_f1:+.3f}) while training {extra_training_time_ratio:.1f}x faster."
        )

    if is_tie:
        summary += f" Notice: The performance difference ({delta_f1:+.3f}) is within fold-to-fold statistical noise."

    return {
        "has_both_tiers": True,
        "best_classic": best_classic_name,
        "best_automl": best_automl_name,
        "delta_f1": delta_f1,
        "delta_roc_auc": delta_roc_auc,
        "extra_training_time_ratio": extra_training_time_ratio,
        "extra_inference_time_ratio": extra_inference_time_ratio,
        "verdict": verdict,
        "is_statistical_tie": is_tie,
        "summary": summary,
    }
