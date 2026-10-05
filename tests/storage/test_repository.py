from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from acbs.storage.models import Base
from acbs.storage.repository import Repository


@pytest.fixture
def repo_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield Repository(session)
    session.close()


def test_repository_crud_flow(repo_session: Repository):
    ds = repo_session.create_dataset(
        name="test_ds",
        source="local",
        n_rows=100,
        n_features=5,
        profile_json={"n_rows": 100},
        file_hash="hash123",
    )
    assert ds.id is not None
    fetched_ds = repo_session.get_dataset_by_hash("hash123")
    assert fetched_ds.name == "test_ds"

    exp = repo_session.create_experiment(
        dataset_id=ds.id,
        config={"models": ["logistic_regression"]},
        preferences={"preset": "balanced"},
        seed=42,
    )
    assert exp.id is not None
    assert exp.status == "queued"

    repo_session.update_status(exp.id, "running")
    updated = repo_session.get_experiment(exp.id)
    assert updated.status == "running"

    result_dict = {
        "seed": 42,
        "models": [
            {
                "model_name": "logistic_regression",
                "status": "ok",
                "test_metrics": {"accuracy": 0.85, "f1_macro": 0.84},
                "training_time": 0.5,
                "prediction_time_per_1k": 0.01,
            }
        ],
    }
    repo_session.save_results(exp.id, result_dict)
    repo_session.save_recommendation(
        exp.id,
        "logistic_regression",
        [{"model_name": "logistic_regression", "rank": 1}],
        {"headline": "Recommended: logistic_regression"},
        100.0,
    )

    exp_final = repo_session.get_experiment(exp.id)
    assert exp_final.recommendation.model_name == "logistic_regression"

    df = repo_session.load_result_as_dataframe(exp.id)
    assert len(df) == 1
    assert df.iloc[0]["model_name"] == "logistic_regression"
    assert df.iloc[0]["accuracy"] == 0.85
