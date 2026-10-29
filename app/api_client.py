"""HTTP client for the Streamlit UI.

Slow-changing endpoints (health, frameworks, cost estimates) are
cached with ``st.cache_data`` so widget interactions do not re-hit the API on
every rerun — that was the main source of UI buffering. ``status`` and result
endpoints are deliberately never cached.
"""

from __future__ import annotations

import io
import os
from typing import Any

import requests
import streamlit as st

BASE = os.getenv("UI_API_BASE", "http://localhost:8000")


class ApiError(RuntimeError):
    pass


def _raise(resp: requests.Response) -> None:
    if resp.status_code >= 400:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise ApiError(f"{resp.status_code}: {detail}")


@st.cache_data(ttl=30, show_spinner=False)
def health_status() -> dict[str, Any]:
    """Cached liveness check; returns {ok, error} instead of raising, so a dead
    API costs at most 2 s once per 30 s instead of blocking every rerun."""
    try:
        r = requests.get(f"{BASE}/health", timeout=2)
        _raise(r)
        return {"ok": True, "error": ""}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def health() -> dict[str, Any]:
    r = requests.get(f"{BASE}/health", timeout=5)
    _raise(r)
    return r.json()


@st.cache_data(ttl=300, show_spinner=False)
def upload(filename: str, content: bytes, target: str | None = None) -> dict[str, Any]:
    """Upload CSV bytes; omit ``target`` and the API auto-detects it."""
    params = {"target": target} if target else {}
    r = requests.post(
        f"{BASE}/datasets",
        params=params,
        files={"file": (filename, io.BytesIO(content), "text/csv")},
        timeout=120,
    )
    _raise(r)
    return r.json()


def start_experiment(payload: dict[str, Any]) -> dict[str, Any]:
    r = requests.post(f"{BASE}/experiments", json=payload, timeout=30)
    _raise(r)
    return r.json()


def status(exp_id: int) -> dict[str, Any]:
    r = requests.get(f"{BASE}/experiments/{exp_id}/status", timeout=5)
    _raise(r)
    return r.json()


def raw_results(exp_id: int) -> dict[str, Any]:
    r = requests.get(f"{BASE}/experiments/{exp_id}/results", timeout=30)
    _raise(r)
    return r.json()


def recommendation(exp_id: int) -> dict[str, Any]:
    r = requests.get(f"{BASE}/experiments/{exp_id}/recommendation", timeout=30)
    _raise(r)
    return r.json()


def advisor(exp_id: int, question: str | None = None) -> dict[str, Any]:
    """Ask the Benchmark Advisor; 503 means no RESPAN_API_KEY is configured."""
    r = requests.post(f"{BASE}/experiments/{exp_id}/advisor", json={"question": question}, timeout=120)
    _raise(r)
    return r.json()


def rerank(exp_id: int, preferences: dict[str, Any]) -> dict[str, Any]:
    r = requests.post(
        f"{BASE}/experiments/{exp_id}/rerank", json={"preferences": preferences}, timeout=30
    )
    _raise(r)
    return r.json()


@st.cache_data(ttl=10, show_spinner=False)
def history() -> list[dict[str, Any]]:
    r = requests.get(f"{BASE}/experiments", timeout=15)
    _raise(r)
    return r.json()


def report_url(exp_id: int, fmt: str = "html") -> str:
    return f"{BASE}/experiments/{exp_id}/report?format={fmt}"


def model_export_url(exp_id: int) -> str:
    return f"{BASE}/experiments/{exp_id}/export-model"


@st.cache_data(ttl=60, show_spinner=False)
def frameworks() -> list[dict[str, Any]]:
    r = requests.get(f"{BASE}/frameworks", timeout=15)
    _raise(r)
    return r.json()


@st.cache_data(ttl=60, show_spinner=False)
def estimate_cost(payload: dict[str, Any]) -> dict[str, Any]:
    r = requests.post(f"{BASE}/frameworks/estimate", json=payload, timeout=15)
    _raise(r)
    return r.json()


def cancel_experiment(exp_id: int) -> dict[str, Any]:
    r = requests.delete(f"{BASE}/experiments/{exp_id}", timeout=15)
    _raise(r)
    return r.json()


def results(exp_id: int) -> dict[str, Any]:
    r = requests.get(f"{BASE}/experiments/{exp_id}/results", timeout=30)
    _raise(r)
    return r.json()
