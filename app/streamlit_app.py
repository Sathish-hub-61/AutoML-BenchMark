"""Auto Classifier Benchmarking System — Streamlit entry point.

All page logic lives under app/views/, shared components under app/components/.
This file is a pure router. (The directory is deliberately NOT called
``pages/``: Streamlit auto-registers ``pages/*.py`` as native sidebar pages,
which produced a second, broken navigation next to this router.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make sure the app directory is on the path so pages/components can import each other.
_APP_DIR = Path(__file__).parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from api_client import health_status  # noqa: E402
from views import export, history, new_benchmark, progress, results  # noqa: E402

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Auto Classifier Benchmark",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
    .stButton > button { border-radius: 8px; font-weight: 600; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
    .stProgress > div > div { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 acbs")
    st.caption("Auto Classifier Benchmarking System")

    health = health_status()
    if health["ok"]:
        st.success("API connected")
    else:
        st.error(
            "API unreachable — start it with "
            f"`uvicorn acbs.api.main:app --port 8000`\n\n{health['error']}"
        )

    st.divider()

    # A page body can request navigation, but widget-backed session keys may
    # not be written after the widget is instantiated — so honour the request
    # here, BEFORE the radio exists in this run.
    _goto = st.session_state.pop("goto_page", None)
    if _goto:
        st.session_state.nav_page = _goto

    page = st.radio(
        "Navigate",
        ["New Benchmark", "Progress", "Results", "History", "Export"],
        key="nav_page",
    )

    st.divider()
    st.markdown(
        """
**Metrics guide**
- **Accuracy** — overall correctness *(misleading when imbalanced)*
- **Precision / Recall / F1** — class-wise quality
- **ROC-AUC** — ranking quality
- **Train / predict time** — efficiency
- **Budget used ratio** — AutoML only: how much of the budget was consumed
        """
    )

# ── Session defaults ──────────────────────────────────────────────────────────
if "experiment_id" not in st.session_state:
    st.session_state.experiment_id = None

# ── Router ────────────────────────────────────────────────────────────────────
if page == "New Benchmark":
    new_benchmark.render()
elif page == "Progress":
    progress.render()
elif page == "Results":
    results.render()
elif page == "History":
    history.render()
else:
    export.render()
