from __future__ import annotations

import zipfile
from pathlib import Path

import joblib

from acbs.benchmark.engine import BenchmarkConfig, run_benchmark
from acbs.data.loader import load_dataset
from acbs.models.export import export_experiment_model
from acbs.storage import Repository, make_session_factory
from acbs.storage.models import ExperimentRow


def test_export_classic_model(tmp_path: Path):
    SessionLocal = make_session_factory()
    with SessionLocal() as session:
        repo = Repository(session)
        ds = load_dataset("breast_cancer")
        ds_row = repo.create_dataset(
            name="breast_cancer",
            source="sklearn",
            n_rows=len(ds.X),
            n_features=len(ds.X.columns),
        )

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


def test_export_automl_model(tmp_path: Path):
    SessionLocal = make_session_factory()
    with SessionLocal() as session:
        repo = Repository(session)
        ds = load_dataset("breast_cancer")
        ds_row = repo.create_dataset(
            name="breast_cancer",
            source="sklearn",
            n_rows=len(ds.X),
            n_features=len(ds.X.columns),
        )

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
