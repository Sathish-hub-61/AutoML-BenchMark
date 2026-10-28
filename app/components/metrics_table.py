"""Metrics table component — coloured DataFrame with kind badges."""

from __future__ import annotations

import pandas as pd
import streamlit as st

_METRIC_COLS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
_TIME_COLS = ["train_time_s", "predict_time_s"]


def render_metrics_table(df: pd.DataFrame) -> None:
    """Render styled leaderboard table. ``df`` rows must have at least ``model_name`` and ``rank``."""
    if df.empty:
        st.info("No results yet.")
        return

    display_cols = (
        ["rank", "model_name", "kind", "framework"]
        + [c for c in _METRIC_COLS if c in df.columns]
        + [c for c in _TIME_COLS if c in df.columns]
        + ["status"]
    )

    display_cols = [c for c in display_cols if c in df.columns]
    view = df[display_cols].copy()

    # Apply colour gradient to metric columns
    metric_present = [c for c in _METRIC_COLS if c in view.columns]

    def _colour_kind(val: str) -> str:
        if val == "automl":
            return "color: #6c63ff; font-weight: bold"
        return "color: #26a69a"

    styled = view.style.background_gradient(subset=metric_present, cmap="RdYlGn", axis=0)
    if "kind" in view.columns:
        styled = styled.map(_colour_kind, subset=["kind"])
    if "status" in view.columns:
        styled = styled.map(
            lambda v: "color: red" if v not in {"ok", None, ""} else "color: green",
            subset=["status"],
        )
    st.dataframe(styled, width="stretch", hide_index=True)
