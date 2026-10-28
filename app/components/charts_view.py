"""Charts view component — all Plotly charts for the Results page tabs."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_CLASSIC_COLOR = "#26a69a"
_AUTOML_COLOR = "#6c63ff"
_METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
    "train_time_s": "Train Time (s)",
    "predict_time_s": "Predict Time (s)",
}


def _kind_colors(df: pd.DataFrame) -> list[str]:
    return [
        _AUTOML_COLOR if k == "automl" else _CLASSIC_COLOR
        for k in df.get("kind", ["classic"] * len(df))
    ]


# ── Tab 1: Overview ──────────────────────────────────────────────────────────


def overview_charts(df: pd.DataFrame) -> None:
    """Grouped bar chart of main metrics + Pareto scatter."""
    if df.empty:
        st.info("No results to display.")
        return

    metric_cols = [
        c for c in ["accuracy", "f1", "roc_auc", "precision", "recall"] if c in df.columns
    ]
    if not metric_cols:
        st.warning("No metric columns found in results.")
        return

    melted = df.melt(
        id_vars=["model_name", "kind"],
        value_vars=metric_cols,
        var_name="metric",
        value_name="score",
    )
    melted["metric"] = melted["metric"].map(_METRIC_LABELS).fillna(melted["metric"])

    fig_bar = px.bar(
        melted,
        x="model_name",
        y="score",
        color="kind",
        facet_col="metric",
        color_discrete_map={"classic": _CLASSIC_COLOR, "automl": _AUTOML_COLOR},
        title="Performance Metrics by Model",
        labels={"model_name": "Model", "score": "Score", "kind": "Tier"},
        height=420,
    )
    fig_bar.update_layout(
        template="plotly_dark",
        plot_bgcolor="#111",
        paper_bgcolor="#111",
        font=dict(family="Inter", size=12),
        legend=dict(orientation="h", y=-0.25),
    )
    fig_bar.update_xaxes(tickangle=30)
    st.plotly_chart(fig_bar, width="stretch")

    # Pareto scatter: accuracy vs train_time_s
    if "accuracy" in df.columns and "train_time_s" in df.columns:
        fig_par = px.scatter(
            df,
            x="train_time_s",
            y="accuracy",
            color="kind",
            symbol="kind",
            size=[8] * len(df),
            text="model_name",
            color_discrete_map={"classic": _CLASSIC_COLOR, "automl": _AUTOML_COLOR},
            log_x=True,
            title="Pareto: Accuracy vs. Training Time",
            labels={"train_time_s": "Training Time (s, log)", "accuracy": "Accuracy"},
            height=420,
        )
        fig_par.update_traces(textposition="top center", marker=dict(size=12, line=dict(width=1)))
        fig_par.update_layout(
            template="plotly_dark",
            plot_bgcolor="#111",
            paper_bgcolor="#111",
            font=dict(family="Inter", size=12),
        )
        st.plotly_chart(fig_par, width="stretch")


# ── Tab 2: Radar ─────────────────────────────────────────────────────────────


def radar_chart(df: pd.DataFrame, top_n: int = 6) -> None:
    """Radar chart for top-N models."""
    radar_cols = [
        c for c in ["accuracy", "f1", "roc_auc", "precision", "recall"] if c in df.columns
    ]
    if not radar_cols or df.empty:
        st.info("Insufficient data for radar chart.")
        return

    candidates = df.nsmallest(top_n, "rank") if "rank" in df.columns else df.head(top_n)
    fig = go.Figure()
    for _, row in candidates.iterrows():
        vals = [row[c] for c in radar_cols]
        vals += [vals[0]]  # close the loop
        cols = radar_cols + [radar_cols[0]]
        color = _AUTOML_COLOR if row.get("kind") == "automl" else _CLASSIC_COLOR
        fig.add_trace(
            go.Scatterpolar(
                r=vals,
                theta=cols,
                fill="toself",
                name=str(row["model_name"]),
                line_color=color,
                opacity=0.7,
            )
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        template="plotly_dark",
        paper_bgcolor="#111",
        font=dict(family="Inter", size=12),
        title=f"Radar — Top {top_n} Candidates",
        height=500,
    )
    st.plotly_chart(fig, width="stretch")


# ── Tab 3: Time & Cost ───────────────────────────────────────────────────────


def time_cost_charts(df: pd.DataFrame) -> None:
    """Training time, budget adherence, inference latency, peak memory."""
    if df.empty:
        return

    c1, c2 = st.columns(2)

    # Train time (log)
    if "train_time_s" in df.columns:
        fig_train = px.bar(
            df.sort_values("train_time_s"),
            x="model_name",
            y="train_time_s",
            color="kind",
            color_discrete_map={"classic": _CLASSIC_COLOR, "automl": _AUTOML_COLOR},
            log_y=True,
            title="Training Time (log scale, seconds)",
            labels={"train_time_s": "Time (s)", "model_name": "Model"},
        )
        fig_train.update_layout(
            template="plotly_dark",
            plot_bgcolor="#111",
            paper_bgcolor="#111",
            font=dict(family="Inter", size=11),
            height=340,
            showlegend=False,
        )
        c1.plotly_chart(fig_train, width="stretch")

    # Inference latency
    if "predict_time_s" in df.columns:
        fig_infer = px.bar(
            df.sort_values("predict_time_s"),
            x="model_name",
            y="predict_time_s",
            color="kind",
            color_discrete_map={"classic": _CLASSIC_COLOR, "automl": _AUTOML_COLOR},
            log_y=True,
            title="Inference Latency / 1k rows (log, s)",
            labels={"predict_time_s": "Time (s)", "model_name": "Model"},
        )
        fig_infer.update_layout(
            template="plotly_dark",
            plot_bgcolor="#111",
            paper_bgcolor="#111",
            font=dict(family="Inter", size=11),
            height=340,
            showlegend=False,
        )
        c2.plotly_chart(fig_infer, width="stretch")

    # Budget adherence for AutoML
    automl_df = (
        df[df.get("kind", pd.Series(dtype=str)) == "automl"]
        if "kind" in df.columns
        else pd.DataFrame()
    )
    if not automl_df.empty and "budget_used_ratio" in automl_df.columns:
        fig_budget = px.bar(
            automl_df,
            x="model_name",
            y="budget_used_ratio",
            color="model_name",
            title="Budget Used Ratio (AutoML only)",
            labels={"budget_used_ratio": "Ratio (0–1)", "model_name": "Framework"},
        )
        fig_budget.add_hline(
            y=1.0, line_dash="dash", line_color="red", annotation_text="Budget limit"
        )
        fig_budget.update_layout(
            template="plotly_dark",
            plot_bgcolor="#111",
            paper_bgcolor="#111",
            font=dict(family="Inter", size=11),
            height=320,
            showlegend=False,
        )
        st.plotly_chart(fig_budget, width="stretch")

    # Peak memory
    if "peak_mem_mb" in df.columns:
        fig_mem = px.bar(
            df.sort_values("peak_mem_mb", ascending=False),
            x="model_name",
            y="peak_mem_mb",
            color="kind",
            color_discrete_map={"classic": _CLASSIC_COLOR, "automl": _AUTOML_COLOR},
            title="Peak Memory (MB)",
            labels={"peak_mem_mb": "Memory (MB)", "model_name": "Model"},
        )
        fig_mem.update_layout(
            template="plotly_dark",
            plot_bgcolor="#111",
            paper_bgcolor="#111",
            font=dict(family="Inter", size=11),
            height=320,
        )
        st.plotly_chart(fig_mem, width="stretch")


# ── Tab 4: Sensitivity tornado ────────────────────────────────────────────────


def sensitivity_chart(score_table: list[dict]) -> None:
    """Tornado chart showing rank-shift when weights change."""
    if not score_table:
        st.info("No score table available for sensitivity analysis.")
        return

    df = pd.DataFrame(score_table)
    if "model_name" not in df.columns or "weighted_score" not in df.columns:
        st.warning("score_table missing required columns.")
        return

    fig = px.bar(
        df.sort_values("weighted_score"),
        x="weighted_score",
        y="model_name",
        orientation="h",
        color="kind" if "kind" in df.columns else None,
        color_discrete_map={"classic": _CLASSIC_COLOR, "automl": _AUTOML_COLOR},
        title="Weighted Score (current weights)",
        labels={"weighted_score": "Score", "model_name": "Model"},
    )
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#111",
        paper_bgcolor="#111",
        font=dict(family="Inter", size=12),
        height=max(300, len(df) * 36),
    )
    st.plotly_chart(fig, width="stretch")


# ── Tab 5: Framework internals ────────────────────────────────────────────────


def framework_internals(df: pd.DataFrame) -> None:
    """Show per-framework leaderboard and model_summary for AutoML rows."""
    automl_df = (
        df[df.get("kind", pd.Series(dtype=str)) == "automl"]
        if "kind" in df.columns
        else pd.DataFrame()
    )
    if automl_df.empty:
        st.info("No AutoML frameworks ran in this experiment.")
        return

    for _, row in automl_df.iterrows():
        with st.expander(f"🔬 {row.get('model_name', 'Unknown')}  —  {row.get('status', '')}"):
            c1, c2 = st.columns(2)
            c1.metric("F1", f"{row.get('f1', 0):.4f}" if row.get("f1") is not None else "—")
            c1.metric(
                "ROC-AUC", f"{row.get('roc_auc', 0):.4f}" if row.get("roc_auc") is not None else "—"
            )
            c2.metric("Train time (s)", f"{row.get('train_time_s', 0):.1f}")
            c2.metric(
                "Budget used",
                f"{row.get('budget_used_ratio', 0):.1%}"
                if row.get("budget_used_ratio") is not None
                else "—",
            )
            if row.get("model_summary"):
                st.code(row["model_summary"], language="text")
            if row.get("n_models_evaluated"):
                st.caption(f"Models evaluated: {row['n_models_evaluated']}")
            if row.get("peak_mem_mb"):
                st.caption(f"Peak memory: {row['peak_mem_mb']:.0f} MB")
