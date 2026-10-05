from __future__ import annotations

import time

from fastapi.testclient import TestClient

from acbs.api.main import app

client = TestClient(app)


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
