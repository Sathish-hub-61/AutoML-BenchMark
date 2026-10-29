from __future__ import annotations

import zipfile
from pathlib import Path

import joblib
import pytest
from sqlalchemy import delete

from acbs.benchmark.engine import BenchmarkConfig, run_benchmark
from acbs.data.loader import LoadedDataset, load_dataset
from acbs.models.export import export_experiment_model
from acbs.storage import Repository, make_session_factory
from acbs.storage.models import DatasetRow, ExperimentRow
from acbs.utils.paths import repo_root


@pytest.fixture()
def uploaded_csv():
    """Remove the CSVs and dataset rows tests store under data/raw, even on failure."""
    created: list[tuple[int, Path]] = []
    yield created
    factory = make_session_factory()
    session = factory()
    try:
        for ds_id, path in created:
            # Core delete: the ORM would null out experiments' NOT NULL FK.
            session.execute(delete(DatasetRow).where(DatasetRow.id == ds_id))
            session.commit()
            path.unlink(missing_ok=True)
    finally:
        session.close()


def _store_as_upload(repo: Repository, ds: LoadedDataset) -> tuple[DatasetRow, Path]:
    """Persist a dataset exactly like the upload route: a raw CSV + an upload row."""
    import uuid

    name = f"export_{uuid.uuid4().hex[:10]}.csv"
    path = repo_root() / "data" / "raw" / name
    frame = ds.X.copy()
    frame["target"] = ds.y.to_numpy()
    frame.to_csv(path, index=False)
    row = repo.create_dataset(
        name=name,
        source="upload",
        n_rows=len(ds.X),
        n_features=len(ds.X.columns),
        profile_json={
            "meta": {
                "target_column": "target",
                "target_source": "explicit",
                "source": "upload",
                "name": name,
            }
        },
    )
    return row, path


def test_export_classic_model(tmp_path: Path, uploaded_csv):
    SessionLocal = make_session_factory()
    with SessionLocal() as session:
        repo = Repository(session)
        ds = load_dataset("breast_cancer")
        ds_row, csv_path = _store_as_upload(repo, ds)
        uploaded_csv.append((ds_row.id, csv_path))

        cfg = BenchmarkConfig(models=["logistic_regression"], tuning="none", cv_folds=2)
        exp = repo.create_experiment(
            dataset_id=ds_row.id,
            config=cfg.model_dump(),
            preferences={"preset": "balanced"},
            seed=42,
        )

        from acbs.benchmark.aggregate import results_to_dataframe

        results = run_benchmark(ds, cfg)
        repo.save_results(exp.id, results.model_dump())
        results_df = results_to_dataframe(results)
        repo.save_recommendation(
            exp.id,
            "logistic_regression",
            results_df.to_dict(orient="records"),
            {"headline": "Logistic Regression is best"},
            85.0,
        )

        exp_row = session.get(ExperimentRow, exp.id)
        zip_path = export_experiment_model(exp_row, output_dir=tmp_path)

        assert zip_path.exists()
        assert zip_path.name.endswith(".zip")

        # Inspect zip contents
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            assert "model.joblib" in names
            assert "metadata.json" in names
            assert "predict_snippet.py" in names
            assert "requirements.txt" in names
            assert "README.md" in names

            # Verify that model.joblib unpickles and predicts
            zf.extract("model.joblib", path=tmp_path)
            loaded_pipe = joblib.load(tmp_path / "model.joblib")
            sample_df = ds.X.head(5)
            preds = loaded_pipe.predict(sample_df)
            assert len(preds) == 5


def test_export_automl_model(tmp_path: Path, uploaded_csv):
    SessionLocal = make_session_factory()
    with SessionLocal() as session:
        repo = Repository(session)
        ds = load_dataset("breast_cancer")
        ds_row, csv_path = _store_as_upload(repo, ds)
        uploaded_csv.append((ds_row.id, csv_path))

        cfg = BenchmarkConfig(models=["logistic_regression"], tuning="none", cv_folds=2)
        exp = repo.create_experiment(
            dataset_id=ds_row.id,
            config=cfg.model_dump(),
            preferences={"preset": "balanced"},
            seed=42,
        )

        # Fake an AutoML recommendation
        repo.save_recommendation(
            exp.id,
            "flaml",
            [],
            {"headline": "FLAML recommended"},
            90.0,
        )

        exp_row = session.get(ExperimentRow, exp.id)
        zip_path = export_experiment_model(exp_row, output_dir=tmp_path)

        assert zip_path.exists()
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            assert "metadata.json" in names
            assert "load_snippet.py" in names
            assert "requirements.txt" in names
            assert "README.md" in names
