from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from acbs.storage.models import DatasetRow, ExperimentRow, ModelRunRow, RecommendationRow
from acbs.utils.jsonsafe import json_safe


class Repository:
    def __init__(self, session: Session):
        self.session = session

    def get_dataset_by_hash(self, file_hash: str) -> DatasetRow | None:
        return self.session.query(DatasetRow).filter_by(file_hash=file_hash).one_or_none()

    def create_dataset(self, **kwargs) -> DatasetRow:
        row = DatasetRow(**kwargs)
        self.session.add(row)
        self.session.commit()
        self.session.refresh(row)
        return row

    def get_dataset(self, dataset_id: int) -> DatasetRow | None:
        return self.session.get(DatasetRow, dataset_id)

    def list_datasets(self) -> list[DatasetRow]:
        return list(self.session.query(DatasetRow).order_by(DatasetRow.id.desc()).all())

    def create_experiment(
        self, dataset_id: int, config: dict, preferences: dict, seed: int
    ) -> ExperimentRow:
        row = ExperimentRow(
            dataset_id=dataset_id,
            config_json=config,
            preferences_json=preferences,
            status="queued",
            seed=seed,
            progress_json={"pct": 0, "current_model": None, "message": "queued"},
        )
        self.session.add(row)
        self.session.commit()
        self.session.refresh(row)
        return row

    def update_status(self, experiment_id: int, status: str, **kwargs) -> ExperimentRow:
        row = self.session.get(ExperimentRow, experiment_id)
        if row is None:
            raise KeyError(experiment_id)
        row.status = status
        for k, v in kwargs.items():
            setattr(row, k, v)
        self.session.commit()
        return row

    def save_results(self, experiment_id: int, result_dict: dict[str, Any]) -> None:
        row = self.session.get(ExperimentRow, experiment_id)
        if row is None:
            raise KeyError(experiment_id)
        # Sanitise before the JSON column: NaN literals are illegal on jsonb.
        result_dict = json_safe(result_dict)
        row.result_json = result_dict
        row.lib_versions_json = result_dict.get("lib_versions")
        row.model_runs.clear()
        for m in result_dict.get("models", []):
            row.model_runs.append(
                ModelRunRow(
                    model_name=m["model_name"],
                    kind=m.get("kind", "classic"),
                    framework=m.get("framework"),
                    framework_version=m.get("framework_version"),
                    budget_s=m.get("budget_s"),
                    budget_used_s=m.get("training_time") if m.get("kind") == "automl" else None,
                    peak_mem_mb=m.get("peak_mem_mb"),
                    model_summary=m.get("model_summary"),
                    status=m.get("status", "ok"),
                    error=m.get("error"),
                    notes=m.get("notes"),
                    cv_metrics_json=m.get("cv_metrics"),
                    test_metrics_json=m.get("test_metrics"),
                    training_time=m.get("training_time"),
                    prediction_time_per_1k=m.get("prediction_time_per_1k"),
                    best_params_json=m.get("best_params"),
                    fold_scores_json=m.get("fold_scores"),
                )
            )
        self.session.commit()

    def save_recommendation(
        self, experiment_id: int, model_name: str, table: Any, explanation: dict, stability: float
    ):
        row = self.session.get(ExperimentRow, experiment_id)
        if row is None:
            raise KeyError(experiment_id)
        rec = row.recommendation or RecommendationRow(
            experiment_id=experiment_id, model_name=model_name
        )
        rec.model_name = model_name
        rec.score_table_json = json_safe(table)
        rec.explanation_json = json_safe(explanation)
        rec.stability_pct = stability if stability == stability else None  # NaN -> NULL
        row.recommendation = rec
        self.session.commit()

    def get_experiment(self, experiment_id: int) -> ExperimentRow | None:
        return self.session.get(ExperimentRow, experiment_id)

    def list_experiments(self) -> list[ExperimentRow]:
        return list(self.session.query(ExperimentRow).order_by(ExperimentRow.id.desc()).all())

    def delete_experiment(self, experiment_id: int) -> None:
        row = self.session.get(ExperimentRow, experiment_id)
        if row:
            self.session.delete(row)
            self.session.commit()

    def load_result_as_dataframe(self, experiment_id: int) -> pd.DataFrame:
        row = self.get_experiment(experiment_id)
        if not row or not row.model_runs:
            return pd.DataFrame()
        records = []
        for m in row.model_runs:
            rec = {
                "model_name": m.model_name,
                "status": m.status,
                "training_time": m.training_time,
                "prediction_time_per_1k": m.prediction_time_per_1k,
                "notes": m.notes,
            }
            if m.test_metrics_json:
                rec.update(m.test_metrics_json)
            records.append(rec)
        return pd.DataFrame(records)
