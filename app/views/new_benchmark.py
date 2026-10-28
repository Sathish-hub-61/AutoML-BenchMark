"""New Benchmark page — upload a CSV and the system runs everything."""

from __future__ import annotations

import streamlit as st

from api_client import ApiError, start_experiment
from components.experiment_form import experiment_form


def render() -> None:
    st.header("🆕 New Benchmark")
    body = experiment_form()
    if body is None:
        return

    try:
        created = start_experiment(body)
    except ApiError as exc:
        st.error(str(exc))
        return

    st.session_state.experiment_id = created["id"]
    frameworks = created.get("automl_frameworks") or []
    budget = created.get("automl_budget_s")

    if (body.get("automl") or {}).get("auto"):
        # Zero-interaction run: jump straight to live progress, and from there
        # straight to the results when it finishes.
        st.session_state.auto_view_results = True
        st.session_state.goto_page = "Progress"
        st.rerun()

    parts = [f"✅ Experiment **#{created['id']}** started!"]
    if frameworks:
        parts.append(f"AutoML: {', '.join(frameworks)} at {budget}s budget each.")
    else:
        parts.append("No AutoML environments installed — running the classic tier.")
    parts.append("Switch to **Progress** to track it.")
    st.success(" ".join(parts))
