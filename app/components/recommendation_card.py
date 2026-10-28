"""Recommendation card component."""

from __future__ import annotations

import streamlit as st

_VERDICT_COLORS = {
    "AutoML worth it": "#43a047",
    "Marginal": "#fb8c00",
    "Not worth it": "#e53935",
}


def recommendation_card(rec: dict) -> None:
    """Render the top recommendation with explanation, stability, and AutoML verdict."""
    expn = rec.get("explanation") or {}
    headline = expn.get("headline") or rec.get("model_name", "—")
    paragraphs = expn.get("paragraphs", [])
    stability = rec.get("stability_pct", 0)
    verdict = expn.get("automl_verdict") or ""

    vc = _VERDICT_COLORS.get(verdict, "#78909c")

    # NOTE: deliberately a single line — markdown-it ends a raw-HTML block at
    # the first line containing a closing tag, so a multi-line card leaks
    # literal "</div>" text onto the page.
    st.markdown(
        f"""<div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 60%,#0f3460 100%);border-radius:16px;padding:28px 32px;margin-bottom:20px;border:1px solid #2a2a4a"><div style="font-size:1.6rem;font-weight:800;color:#e0e0ff;margin-bottom:6px">🏆 {headline}</div><div style="display:flex;gap:16px;margin-top:8px;flex-wrap:wrap"><span style="background:#1e3a5f;color:#90caf9;padding:4px 12px;border-radius:20px;font-size:0.85em">Stability: {stability:.0f}%</span>{f'<span style="background:{vc}22;color:{vc};border:1px solid {vc};padding:4px 12px;border-radius:20px;font-size:0.85em">{verdict}</span>' if verdict else ""}</div></div>""",
        unsafe_allow_html=True,
    )

    if paragraphs:
        with st.expander("📋 Full Explanation", expanded=True):
            for para in paragraphs:
                st.markdown(para)

    # Value analysis block
    va = expn.get("automl_value_analysis")
    if va:
        c1, c2, c3 = st.columns(3)
        c1.metric("Best AutoML F1 Δ over classic", f"{va.get('delta_f1', 0):+.4f}")
        c2.metric(
            "Training time ratio (AutoML/classic)", f"{va.get('extra_training_time_ratio', 0):.1f}×"
        )
        c3.metric("Inference time ratio", f"{va.get('extra_inference_time_ratio', 0):.1f}×")
