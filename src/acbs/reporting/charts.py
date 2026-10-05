from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import auc, confusion_matrix, roc_curve

PALETTE = ["#2563eb", "#16a34a", "#d97706", "#dc2626", "#7c3aed", "#0891b2", "#4b5563", "#db2777"]


def performance_bar(df: pd.DataFrame, metric: str = "f1_macro") -> go.Figure:
    ok = df[df.get("status", "ok") == "ok"]
    fig = px.bar(ok, x="model_name", y=metric, color="model_name", color_discrete_sequence=PALETTE)
    fig.update_layout(showlegend=False, title=f"{metric} by model")
    return fig


def grouped_metrics_bar(df: pd.DataFrame, metrics: list[str] | None = None) -> go.Figure:
    metrics = metrics or ["accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc"]
    ok = df[df.get("status", "ok") == "ok"]
    long = ok.melt(id_vars=["model_name"], value_vars=[m for m in metrics if m in ok.columns])
    return px.bar(long, x="model_name", y="value", color="variable", barmode="group", color_discrete_sequence=PALETTE)


def metric_radar(df: pd.DataFrame, models: list[str], metrics: list[str]) -> go.Figure:
    ok = df[df["model_name"].isin(models)]
    fig = go.Figure()
    for _, row in ok.iterrows():
        vals = []
        for m in metrics:
            col = ok[m] if m in ok.columns else pd.Series([0])
            lo, hi = col.min(), col.max()
            v = row.get(m, 0) or 0
            vals.append(0.5 if hi == lo else (v - lo) / (hi - lo))
        fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=metrics + metrics[:1], fill="toself", name=row["model_name"]))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title="Normalised metrics")
    return fig


def time_comparison(df: pd.DataFrame) -> go.Figure:
    ok = df[df.get("status", "ok") == "ok"]
    long = ok.melt(id_vars=["model_name"], value_vars=["training_time", "prediction_time_per_1k"])
    fig = px.bar(long, x="model_name", y="value", color="variable", barmode="group", log_y=True, color_discrete_sequence=PALETTE)
    fig.update_layout(title="Training vs prediction time (log y)")
    return fig


def accuracy_vs_time_scatter(df: pd.DataFrame, pareto: list[str] | None = None) -> go.Figure:
    ok = df[df.get("status", "ok") == "ok"].copy()
    ok["on_pareto"] = ok["model_name"].isin(pareto or [])
    fig = px.scatter(
        ok,
        x="training_time",
        y="accuracy",
        size="f1_macro" if "f1_macro" in ok.columns else None,
        color="on_pareto",
        hover_name="model_name",
        color_discrete_sequence=["#9ca3af", "#2563eb"],
        title="Accuracy vs training time (Pareto highlighted)",
    )
    return fig


def cv_boxplot(fold_scores: pd.DataFrame) -> go.Figure:
    long = fold_scores.melt(var_name="model_name", value_name="score")
    return px.box(long, x="model_name", y="score", color="model_name", color_discrete_sequence=PALETTE, title="CV fold variance")


def confusion_matrix_heatmap(y_true, y_pred, labels=None) -> go.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title="Confusion matrix")
    return fig


def roc_curves(y_true, y_proba) -> go.Figure:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    fig = go.Figure()
    if y_proba.ndim == 1 or y_proba.shape[1] == 2:
        scores = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
        fpr, tpr, _ = roc_curve(y_true, scores)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC={auc(fpr, tpr):.3f}"))
    fig.update_layout(title="ROC", xaxis_title="FPR", yaxis_title="TPR")
    return fig


def score_breakdown_stacked_bar(scored_df: pd.DataFrame, weight_cols: list[str] | None = None) -> go.Figure:
    ok = scored_df[~scored_df.get("excluded", False)]
    fig = px.bar(ok, x="model_name", y="composite_score", color="model_name", color_discrete_sequence=PALETTE)
    fig.update_layout(title="Composite recommendation score", showlegend=False)
    return fig
