"""Export page — report & model bundle download links."""
from __future__ import annotations

import streamlit as st

from api_client import model_export_url, report_url


def render() -> None:
    st.header("📥 Export")
    exp_id = int(st.session_state.get("experiment_id") or 1)
    exp_override = st.number_input("Experiment ID", min_value=1, value=exp_id, key="export_exp_id")
    eid = int(exp_override)

    st.subheader("📊 Reports & Data")
    st.markdown("Download experiment report in your preferred format:")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"[📄 HTML Report]({report_url(eid, 'html')})", unsafe_allow_html=False)
    c2.markdown(f"[📊 CSV]({report_url(eid, 'csv')})", unsafe_allow_html=False)
    c3.markdown(f"[🗂 JSON]({report_url(eid, 'json')})", unsafe_allow_html=False)
    c4.markdown(f"[🖨 PDF]({report_url(eid, 'pdf')})", unsafe_allow_html=False)

    st.divider()
    st.subheader("📦 Model Artifact Export")
    st.markdown(
        """
        Download the **recommended winning model** packaged into a deployable bundle (`.zip`):
        - **Classic models**: Scikit-learn Pipeline (preprocessing + classifier) refit on all data, `model.joblib`, `predict_snippet.py`, and `requirements.txt`.
        - **AutoML contenders**: Native loading snippet, framework version metadata, and environment instructions.
        """
    )
    st.markdown(
        f"""<div style="margin-top:10px">
        <a href="{model_export_url(eid)}" target="_blank" style="
            display:inline-block;
            background-color:#6c63ff;
            color:white;
            padding:10px 20px;
            border-radius:8px;
            text-decoration:none;
            font-weight:700;">
            ⬇️ Download Trained Model Bundle (.zip)
        </a>
        </div>""",
        unsafe_allow_html=True,
    )
