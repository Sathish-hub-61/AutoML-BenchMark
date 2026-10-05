from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, model_validator

BENEFIT = ["accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc", "interpretability_score"]
COST = ["training_time", "prediction_time_per_1k"]

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
    },
    "fast_inference": {
        "accuracy": 0.1,
        "f1": 0.15,
        "precision": 0.05,
        "recall": 0.05,
        "roc_auc": 0.05,
        "train_time": 0.2,
        "predict_time": 0.35,
        "interpretability": 0.05,
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
        ) = [float(x) for x in weights]
        return self

    def metric_weights(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "f1_macro": self.f1,
            "precision_macro": self.precision,
            "recall_macro": self.recall,
            "roc_auc": self.roc_auc,
            "training_time": self.train_time,
            "prediction_time_per_1k": self.predict_time,
            "interpretability_score": self.interpretability,
        }


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
    mapping_present = [c for c in BENEFIT if c in df.columns]
    for col in mapping_present:
        out[col] = _minmax(df[col].fillna(df[col].median() if df[col].notna().any() else 0), invert=False)
    for col in COST:
        if col in df.columns:
            out[col] = _minmax(df[col].fillna(df[col].max() if df[col].notna().any() else 1), invert=True)
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
        if prefs.max_prediction_time_ms_per_1k is not None and pd.notna(row.get("prediction_time_per_1k")):
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


def pareto_front(results_df: pd.DataFrame, x: str = "training_time", y: str = "f1_macro") -> list[str]:
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


def sensitivity_analysis(results_df: pd.DataFrame, prefs: UserPreferences, *, n: int = 40) -> dict[str, float]:
    rng = np.random.default_rng(42)
    counts: dict[str, int] = {}
    fields = ["accuracy", "f1", "precision", "recall", "roc_auc", "train_time", "predict_time", "interpretability"]
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
