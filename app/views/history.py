"""History page — past experiments table with quick-load."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from api_client import ApiError, history


def render() -> None:
    st.header("📜 History")
    try:
        rows = history()
    except ApiError as exc:
        st.error(str(exc))
        return

    if not rows:
        st.info("No experiments yet. Run a benchmark first.")
        return

    df = pd.DataFrame(rows)

    # Colour-code status column
    def _status_bg(val: str) -> str:
        return {
            "completed": "background-color: #e8f5e9; color: #2e7d32",
            "done": "background-color: #e8f5e9; color: #2e7d32",
            "failed": "background-color: #fce4ec; color: #b71c1c",
            "running": "background-color: #e3f2fd; color: #1565c0",
            "queued": "background-color: #fff8e1; color: #f57f17",
            "cancelling": "background-color: #fff3e0; color: #e65100",
            "cancelled": "background-color: #fff3e0; color: #e65100",
            "timeout": "background-color: #fff3e0; color: #e65100",
        }.get(str(val).lower(), "")

    styled = df.style.map(_status_bg, subset=["status"] if "status" in df.columns else [])
    st.dataframe(styled, width="stretch", hide_index=True)

    pick = st.number_input(
        "Open experiment ID", min_value=1, value=int(rows[0].get("id", 1)), key="hist_pick"
    )
    if st.button("Open →", key="hist_open_btn"):
        st.session_state.experiment_id = int(pick)
        st.success(f"Loaded experiment #{pick}. Switch to **Results**.")
