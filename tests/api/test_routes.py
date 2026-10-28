from __future__ import annotations

import time
import uuid

import pytest
from fastapi.testclient import TestClient

from acbs.api.deps import SessionLocal
from acbs.api.main import app
from acbs.storage.repository import Repository
from acbs.utils.paths import repo_root

client = TestClient(app)


def _create_experiment(
    dataset_id: int, config: dict | None = None, preferences: dict | None = None
) -> int:
    payload = {
        "dataset_id": dataset_id,
        "config": config or {"models": ["logistic_regression"], "cv_folds": 2, "tuning": "none"},
        "preferences": preferences or {"preset": "balanced"},
    }
    resp = client.post("/experiments", json=payload)
    assert resp.status_code == 202, resp.text
    return resp.json()["id"]


def _wait_for(experiment_id: int, timeout_s: int = 60) -> dict:
    for _ in range(timeout_s):
        st = client.get(f"/experiments/{experiment_id}/status").json()
        if st["status"] in {"completed", "failed", "cancelled"}:
            return st
        time.sleep(1)
    raise AssertionError(f"experiment {experiment_id} did not finish: {st}")


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_registry_datasets():
    response = client.get("/datasets/registry")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0


def test_create_and_run_experiment():
    # Register dataset first or get registry dataset
    reg = client.get("/datasets/registry").json()
    bc_name = next(d["name"] for d in reg if d["name"] == "breast_cancer")

    # Post to /datasets/from-registry/{name}
    ds_resp = client.post(f"/datasets/from-registry/{bc_name}")
    assert ds_resp.status_code == 200
    ds_id = ds_resp.json()["id"]

    # Create experiment with 2 fast models
    exp_payload = {
        "dataset_id": ds_id,
        "config": {
            "models": ["logistic_regression", "naive_bayes"],
            "cv_folds": 2,
            "seed": 42,
            "tuning": "none",
        },
        "preferences": {"preset": "balanced"},
    }

    create_resp = client.post("/experiments", json=exp_payload)
    assert create_resp.status_code == 202
    exp_id = create_resp.json()["id"]

    # Poll status
    for _ in range(30):
        st = client.get(f"/experiments/{exp_id}/status").json()
        if st["status"] in {"completed", "failed"}:
            break
        time.sleep(1)

    assert st["status"] == "completed"

    # Get results
    res_resp = client.get(f"/experiments/{exp_id}/results")
    assert res_resp.status_code == 200

    # Get recommendation
    rec_resp = client.get(f"/experiments/{exp_id}/recommendation")
    assert rec_resp.status_code == 200
    assert "model_name" in rec_resp.json()

    # Rerank
    rerank_resp = client.post(
        f"/experiments/{exp_id}/rerank",
        json={"preferences": {"preset": "fast_inference"}},
    )
    assert rerank_resp.status_code == 200

    # Report HTML
    rep_resp = client.get(f"/experiments/{exp_id}/report?format=html")
    assert rep_resp.status_code == 200
    assert "text/html" in rep_resp.headers["content-type"]

    # Model export zip
    export_resp = client.get(f"/experiments/{exp_id}/export-model")
    assert export_resp.status_code == 200
    assert "application/zip" in export_resp.headers["content-type"]
    assert len(export_resp.content) > 100


def test_upload_dataset_runs_end_to_end():
    """Uploaded CSVs must keep their target column, or experiments can never start."""
    name = f"upload_{uuid.uuid4().hex[:10]}.csv"
    path = repo_root() / "data" / "raw" / name
    rows = ["a,b,target"]
    for i in range(60):
        label = "yes" if i % 2 else "no"
        rows.append(f"{i % 7},{i * 0.5},{label}")
    content = "\n".join(rows) + "\n"

    try:
        up = client.post(
            "/datasets",
            params={"target": "target"},
            files={"file": (name, content, "text/csv")},
        )
        assert up.status_code == 200, up.text
        ds_id = up.json()["id"]
        assert path.exists()

        exp_id = _create_experiment(ds_id)
        st = _wait_for(exp_id)
        assert st["status"] == "completed", st.get("error")

        # status payload must carry elapsed time and per-framework state
        assert isinstance(st["elapsed_time_s"], (int, float))
        fw = st["per_framework_status"]
        assert "logistic_regression" in fw
        assert fw["logistic_regression"]["status"] == "ok"
        assert fw["logistic_regression"]["elapsed_s"] is not None

        assert client.get(f"/experiments/{exp_id}/results").status_code == 200
        assert client.get(f"/experiments/{exp_id}/recommendation").status_code == 200
    finally:
        path.unlink(missing_ok=True)


def test_recommendation_failure_keeps_results_available():
    """A constraint that excludes every model must not hide saved results."""
    reg = client.get("/datasets/registry").json()
    ds_id = client.post("/datasets/from-registry/breast_cancer").json()["id"]
    assert reg

    exp_id = _create_experiment(ds_id, preferences={"max_training_time_s": 0.0})
    st = _wait_for(exp_id)

    assert st["status"] == "completed", st
    assert st["error"] and "recommendation failed" in st["error"]
    assert client.get(f"/experiments/{exp_id}/results").status_code == 200
    # no recommendation was stored, so the endpoint 404s rather than crashing
    assert client.get(f"/experiments/{exp_id}/recommendation").status_code == 404


def test_advisor_requires_a_key_and_returns_grounded_narrative(
    monkeypatch: pytest.MonkeyPatch,
):
    """The advisor is optional: no key answers 503, a key answers with the model output."""
    from acbs.recommend import advisor

    ds_id = client.post("/datasets/from-registry/breast_cancer").json()["id"]
    exp_id = _create_experiment(ds_id)
    st = _wait_for(exp_id)
    assert st["status"] == "completed", st.get("error")
    assert client.get(f"/experiments/{exp_id}/recommendation").status_code == 200

    monkeypatch.delenv("RESPAN_API_KEY", raising=False)
    no_key = client.post(f"/experiments/{exp_id}/advisor", json={})
    assert no_key.status_code == 503
    assert "RESPAN_API_KEY" in no_key.json()["detail"]

    calls: dict = {}

    def fake_narrate(explanation, *, question=None, timeout_s=30.0):
        calls["explanation"] = explanation
        calls["question"] = question
        return "Random Forest is recommended because it leads on F1-macro."

    monkeypatch.setenv("RESPAN_API_KEY", "test-key")
    monkeypatch.setattr(advisor, "narrate", fake_narrate)
    ok = client.post(f"/experiments/{exp_id}/advisor", json={"question": "Why?"})
    assert ok.status_code == 200, ok.text
    assert ok.json()["narrative"].startswith("Random Forest is recommended")
    assert ok.json()["model"]
    assert calls["question"] == "Why?"
    assert calls["explanation"]["headline"]  # stored explanation reached the advisor

    # no recommendation stored -> 404, never a gateway call
    assert client.post("/experiments/999999/advisor", json={}).status_code == 404


def test_delete_cancels_running_experiment():
    ds_id = client.post("/datasets/from-registry/wine").json()["id"]
    exp_id = _create_experiment(ds_id)
    _wait_for(exp_id)  # background task has finished; simulate an in-flight run

    session = SessionLocal()
    try:
        Repository(session).update_status(exp_id, "running")
    finally:
        session.close()

    resp = client.delete(f"/experiments/{exp_id}")
    assert resp.status_code == 200
    assert resp.json()["action"] == "cancelling"
    assert client.get(f"/experiments/{exp_id}/status").json()["status"] == "cancelling"

    # a finished experiment is still deletable once it leaves the running states
    session = SessionLocal()
    try:
        Repository(session).update_status(exp_id, "cancelled")
    finally:
        session.close()
    assert client.delete(f"/experiments/{exp_id}").json()["action"] == "deleted"


def test_automl_block_from_ui_is_stored_in_config():
    """The UI posts frameworks under `automl`; the engine reads config.automl_frameworks."""
    ds_id = client.post("/datasets/from-registry/breast_cancer").json()["id"]
    resp = client.post(
        "/experiments",
        json={
            "dataset_id": ds_id,
            "config": {"models": ["logistic_regression"], "cv_folds": 2, "tuning": "none"},
            "preferences": {"preset": "balanced"},
            # unknown id -> worker venv missing -> fast, recorded failure
            "automl": {"frameworks": ["not_a_real_framework"], "budget_s": 5, "n_jobs": 2},
        },
    )
    assert resp.status_code == 202, resp.text
    exp_id = resp.json()["id"]
    st = _wait_for(exp_id)
    assert st["status"] == "completed", st.get("error")
    assert "not_a_real_framework" in st["per_framework_status"]

    results = client.get(f"/experiments/{exp_id}/results").json()
    cfg = results["config"]
    assert cfg["automl_frameworks"] == ["not_a_real_framework"]
    assert cfg["automl_budget_s"] == 5
    assert cfg["automl_n_jobs"] == 2


def test_frameworks_and_estimate_endpoints():
    fw_resp = client.get("/frameworks")
    assert fw_resp.status_code == 200
    fws = fw_resp.json()
    assert isinstance(fws, list)
    assert any(f["id"] == "flaml" for f in fws)

    est_resp = client.post(
        "/frameworks/estimate",
        json={"automl_frameworks": ["flaml", "autogluon"], "automl_budget_s": 60},
    )
    assert est_resp.status_code == 200
    est = est_resp.json()
    assert "total_wall_seconds" in est
    assert est["n_frameworks"] == 2


def test_upload_without_target_autodetects_and_runs():
    """The human only uploads the file: the target is inferred, then a run works end to end."""
    name = f"auto_{uuid.uuid4().hex[:10]}.csv"
    path = repo_root() / "data" / "raw" / name
    rows = ["a,b,label"]
    for i in range(60):
        rows.append(f"{i % 7},{i * 0.5},{'yes' if i % 2 else 'no'}")
    content = "\n".join(rows) + "\n"

    try:
        up = client.post("/datasets", files={"file": (name, content, "text/csv")})
        assert up.status_code == 200, up.text
        body = up.json()
        assert body["target"] == "label"
        assert body["target_source"] == "auto"
        assert body["profile"]["meta"]["target_column"] == "label"

        exp_id = _create_experiment(
            body["id"], config={"models": ["logistic_regression"], "cv_folds": 2, "tuning": "none"}
        )
        st = _wait_for(exp_id)
        assert st["status"] == "completed", st.get("error")
        assert client.get(f"/experiments/{exp_id}/results").status_code == 200
    finally:
        path.unlink(missing_ok=True)


def test_auto_framework_selection_stored_in_config():
    """automl.auto=true: the server picks frameworks — no human choice involved."""
    from unittest.mock import patch

    fake_avail = [
        {
            "id": "fake_single",
            "name": "Fake Single",
            "strategy": "test",
            "kind": "single_pipeline",
            "installed": True,
            "version": "1.0",
            "os_supported": True,
            "reason": None,
        },
        {
            "id": "fake_stacked",
            "name": "Fake Stacked",
            "strategy": "test",
            "kind": "stacked_ensemble",
            "installed": True,
            "version": "1.0",
            "os_supported": True,
            "reason": None,
        },
        {
            "id": "fake_missing",
            "name": "Fake Missing",
            "strategy": "test",
            "kind": "single_pipeline",
            "installed": False,
            "version": None,
            "os_supported": True,
            "reason": "not installed",
        },
    ]

    ds_id = client.post("/datasets/from-registry/breast_cancer").json()["id"]
    with patch("acbs.automl.selection.framework_availability", return_value=fake_avail):
        resp = client.post(
            "/experiments",
            json={
                "dataset_id": ds_id,
                "config": {"models": ["logistic_regression"], "cv_folds": 2, "tuning": "none"},
                "preferences": {"preset": "balanced"},
                "automl": {"auto": True},
            },
        )
    assert resp.status_code == 202, resp.text
    created = resp.json()

    # the response tells the UI exactly what the system chose
    assert created["automl_auto"] is True
    assert created["automl_frameworks"] == ["fake_single", "fake_stacked"]
    assert created["automl_budget_s"] == 60  # small dataset -> policy budget
    assert created["automl_selection"]["selected"]

    st = _wait_for(created["id"])
    assert st["status"] == "completed", st.get("error")

    results = client.get(f"/experiments/{created['id']}/results").json()
    cfg = results["config"]
    assert cfg["automl_frameworks"] == ["fake_single", "fake_stacked"]
    assert cfg["automl_auto"] is True
    assert cfg["automl_selection"]["skipped"]["fake_missing"] == "environment not installed"

    # unavailable framework is never run
    names = {m["model_name"] for m in results["models"]}
    assert "fake_missing" not in names

    # failed frameworks leave NaN metrics in the score table — that must
    # serialise to JSON, not 500 the Results page
    rec = client.get(f"/experiments/{created['id']}/recommendation")
    assert rec.status_code == 200, rec.text
