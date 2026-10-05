from __future__ import annotations

import os
from typing import Any

import requests

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


def health() -> dict[str, Any]:
    r = requests.get(f"{BASE}/health", timeout=5)
    _raise(r)
    return r.json()


def registry() -> list[dict[str, Any]]:
    r = requests.get(f"{BASE}/datasets/registry", timeout=15)
    _raise(r)
    return r.json()


def from_registry(name: str) -> dict[str, Any]:
    r = requests.post(f"{BASE}/datasets/from-registry/{name}", timeout=120)
    _raise(r)
    return r.json()


def upload(path: str, target: str) -> dict[str, Any]:
    with open(path, "rb") as fh:
        r = requests.post(
            f"{BASE}/datasets",
            params={"target": target},
            files={"file": (os.path.basename(path), fh, "text/csv")},
            timeout=120,
        )
    _raise(r)
    return r.json()


def start_experiment(payload: dict[str, Any]) -> dict[str, Any]:
    r = requests.post(f"{BASE}/experiments", json=payload, timeout=30)
    _raise(r)
    return r.json()


def status(exp_id: int) -> dict[str, Any]:
    r = requests.get(f"{BASE}/experiments/{exp_id}/status", timeout=15)
    _raise(r)
    return r.json()


def results(exp_id: int) -> dict[str, Any]:
    r = requests.get(f"{BASE}/experiments/{exp_id}/results", timeout=30)
    _raise(r)
    return r.json()


def recommendation(exp_id: int) -> dict[str, Any]:
    r = requests.get(f"{BASE}/experiments/{exp_id}/recommendation", timeout=30)
    _raise(r)
    return r.json()


def rerank(exp_id: int, preferences: dict[str, Any]) -> dict[str, Any]:
    r = requests.post(f"{BASE}/experiments/{exp_id}/rerank", json={"preferences": preferences}, timeout=30)
    _raise(r)
    return r.json()


def history() -> list[dict[str, Any]]:
    r = requests.get(f"{BASE}/experiments", timeout=15)
    _raise(r)
    return r.json()


def report_url(exp_id: int, fmt: str = "html") -> str:
    return f"{BASE}/experiments/{exp_id}/report?format={fmt}"
