"""Progress page — live per-framework status with non-blocking auto-refresh.

Polling runs inside an ``st.fragment`` every 2 s: only this region reruns, so
the sidebar and page chrome are not re-executed, and there is no
``sleep(3) + rerun`` freezing the whole app anymore.
"""

from __future__ import annotations

from datetime import timedelta

import streamlit as st

from api_client import ApiError, cancel_experiment, status

_TERMINAL = {"completed", "failed", "cancelled"}


def _status_colour(s: str) -> str:
    return {
        "ok": "🟢",
        "completed": "🟢",
        "running": "🔵",
        "queued": "⏳",
        "failed": "🔴",
        "timeout": "🟠",
        "unavailable": "⬛",
        "unsupported": "⬛",
        "skipped": "⚪",
        "cancelling": "🛑",
        "cancelled": "🛑",
    }.get(s, "⚪")


def _sync_exp_id() -> None:
    st.session_state.experiment_id = int(st.session_state.prog_exp_id)


def render() -> None:
    st.header("📊 Progress")

    # Keep the manual ID input in step with the experiment started elsewhere.
    if st.session_state.get("experiment_id"):
        wanted = int(st.session_state.experiment_id)
        if st.session_state.get("prog_exp_id") != wanted:
            st.session_state.prog_exp_id = wanted

    st.number_input(
        "Experiment ID",
        min_value=1,
        value=int(st.session_state.get("experiment_id") or 1),
        key="prog_exp_id",
        on_change=_sync_exp_id,
    )

    if not st.session_state.get("experiment_id"):
        st.info("No experiment yet — start one on the **New Benchmark** page.")
        return

    _live(int(st.session_state.experiment_id))


@st.fragment(run_every=timedelta(seconds=2))
def _live(exp_id: int) -> None:
    try:
        s = status(exp_id)
    except ApiError as exc:
        st.error(str(exc))
        return

    prog = s.get("progress") or {}
    pct = min(int(prog.get("pct") or 0), 100)
    msg = prog.get("message") or s.get("status", "")

    st.progress(pct, text=f"{pct}% — {msg}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Status", s.get("status", "—"))
    elapsed = s.get("elapsed_time_s")
    c2.metric(
        "Elapsed (s)",
        f"{elapsed:.0f}" if isinstance(elapsed, (int, float)) else "—",
    )
    c3.metric("Experiment", s.get("id", "—"))

    if s.get("error"):
        st.warning(s["error"])

    # Per-framework status: {name: {status, elapsed_s, budget_s}}
    fw_statuses = s.get("per_framework_status") or {}
    if fw_statuses:
        st.subheader("Model Status")
        for name, info in fw_statuses.items():
            info = info or {}
            fw_status = str(info.get("status", "?"))
            icon = _status_colour(fw_status)
            st.markdown(f"**{icon} {name}** — {fw_status}")
            fw_elapsed = info.get("elapsed_s")
            budget = info.get("budget_s")
            if budget:
                ratio = min(float(fw_elapsed or 0) / float(budget), 1.0)
                st.progress(
                    ratio,
                    text=f"{float(fw_elapsed or 0):.0f}s / {float(budget):.0f}s budget",
                )
            elif fw_elapsed is not None:
                st.caption(f"{float(fw_elapsed):.2f}s")
    else:
        st.json(s)

    state = s.get("status")
    if state in {"queued", "running", "cancelling"}:
        if st.button("🛑 Cancel", key="prog_cancel", type="secondary"):
            try:
                cancel_experiment(exp_id)
                st.warning("Cancellation requested — the run stops after the current model.")
            except ApiError as exc:
                st.error(str(exc))
        st.caption("Auto-refreshes every 2 s.")
    else:
        auto_jump = st.session_state.pop("auto_view_results", False)
        if auto_jump and state == "completed":
            # Fully hands-off run: show the results as soon as they exist.
            st.session_state.experiment_id = exp_id
            st.session_state.goto_page = "Results"
            st.rerun()
        st.caption("Run finished — open the **Results** page for the recommendation.")
