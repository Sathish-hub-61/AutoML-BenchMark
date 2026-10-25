from __future__ import annotations

import threading

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from sqlalchemy.orm import Session

from acbs.api.deps import SessionLocal, get_session
from acbs.api.schemas import ExperimentCreate, ExperimentStatus, RerankRequest
from acbs.benchmark.aggregate import results_to_dataframe
from acbs.benchmark.engine import BenchmarkResult, run_benchmark
from acbs.data.loader import LoadedDataset, load_dataset, load_local
from acbs.recommend.explain import generate_explanation
from acbs.recommend.scoring import UserPreferences, score_models, sensitivity_analysis
from acbs.reporting.report import build_html_report, export_csv, export_pdf
from acbs.storage.models import DatasetRow
from acbs.storage.repository import Repository
from acbs.utils.logging import get_logger
from acbs.utils.paths import repo_root
from acbs.utils.time import utc_now

logger = get_logger(__name__)
router = APIRouter(prefix="/experiments", tags=["experiments"])
_run_lock = threading.Lock()


def _loaded_from_row(row: DatasetRow) -> LoadedDataset:
    meta = (row.profile_json or {}).get("meta") or {}
    target = meta.get("target_column")
    if row.source == "upload":
        path = repo_root() / "data" / "raw" / row.name
        if not target:
            raise ValueError("uploaded dataset missing target")
        return load_local(path, target)
    return load_dataset(row.name, target)


def _recommend(result: BenchmarkResult, prefs: UserPreferences):
    df = results_to_dataframe(result)
    scored = score_models(df, prefs)
    eligible = scored[~scored["excluded"]]
    if eligible.empty:
        raise ValueError("No models satisfied the constraints")
    rec = eligible.sort_values("rank").iloc[0]
    stability = sensitivity_analysis(df, prefs)
    explanation = generate_explanation(rec, df, result.profile, prefs, stability)
    table = scored.drop(columns=["best_params"], errors="ignore").to_dict(orient="records")
    return rec, table, explanation, stability.get(rec["model_name"], 0.0) * 100, scored


def _execute(experiment_id: int) -> None:
    session = SessionLocal()
    repo = Repository(session)
    try:
        exp = repo.get_experiment(experiment_id)
        if exp is None:
            return
        repo.update_status(experiment_id, "running", started_at=utc_now())
        dataset = _loaded_from_row(exp.dataset)
        cfg = dict(exp.config_json or {})

        def progress(model_name, step, total, message):
            pct = int(100 * step / max(total, 1))
            s = SessionLocal()
            try:
                Repository(s).update_status(
                    experiment_id,
                    "running",
                    progress_json={"pct": pct, "current_model": model_name, "message": message},
                )
            finally:
                s.close()

        result = run_benchmark(dataset, cfg, progress_callback=progress)
        repo.save_results(experiment_id, result.model_dump())
        prefs = UserPreferences.model_validate(exp.preferences_json or {})
        rec, table, explanation, stab, _ = _recommend(result, prefs)
        repo.save_recommendation(experiment_id, rec["model_name"], table, explanation, stab)
        repo.update_status(
            experiment_id,
            "completed",
            finished_at=utc_now(),
            progress_json={"pct": 100, "current_model": rec["model_name"], "message": "done"},
        )
    except Exception as exc:
        logger.exception("experiment %s failed", experiment_id)
        repo.update_status(experiment_id, "failed", error=str(exc), finished_at=utc_now())

    finally:
        session.close()


@router.post("", status_code=202)
def create_experiment(payload: ExperimentCreate, background: BackgroundTasks, session: Session = Depends(get_session)):
    repo = Repository(session)
    ds = repo.get_dataset(payload.dataset_id)
    if not ds:
        raise HTTPException(404, "dataset not found")
    exp = repo.create_experiment(
        payload.dataset_id,
        payload.config.model_dump(),
        payload.preferences.model_dump(),
        payload.config.seed,
    )
    background.add_task(_execute, exp.id)
    return {"id": exp.id, "status": exp.status}


@router.get("/{experiment_id}/status", response_model=ExperimentStatus)
def status(experiment_id: int, session: Session = Depends(get_session)):
    exp = Repository(session).get_experiment(experiment_id)
    if not exp:
        raise HTTPException(404, "experiment not found")
    return ExperimentStatus(id=exp.id, status=exp.status, progress=exp.progress_json, error=exp.error)


@router.get("/{experiment_id}/results")
def results(experiment_id: int, session: Session = Depends(get_session)):
    exp = Repository(session).get_experiment(experiment_id)
    if not exp:
        raise HTTPException(404, "experiment not found")
    if exp.status != "completed":
        raise HTTPException(409, f"experiment status is {exp.status}")
    return exp.result_json


@router.get("/{experiment_id}/recommendation")
def recommendation(experiment_id: int, session: Session = Depends(get_session)):
    exp = Repository(session).get_experiment(experiment_id)
    if not exp or not exp.recommendation:
        raise HTTPException(404, "recommendation not found")
    rec = exp.recommendation
    return {
        "model_name": rec.model_name,
        "explanation": rec.explanation_json,
        "stability_pct": rec.stability_pct,
        "score_table": rec.score_table_json,
    }


@router.post("/{experiment_id}/rerank")
def rerank(experiment_id: int, payload: RerankRequest, session: Session = Depends(get_session)):
    repo = Repository(session)
    exp = repo.get_experiment(experiment_id)
    if not exp or not exp.result_json:
        raise HTTPException(404, "results not found")
    result = BenchmarkResult.model_validate(exp.result_json)
    rec, table, explanation, stab, _ = _recommend(result, payload.preferences)
    repo.save_recommendation(experiment_id, rec["model_name"], table, explanation, stab)
    repo.update_status(experiment_id, exp.status, preferences_json=payload.preferences.model_dump())
    return {
        "model_name": rec["model_name"],
        "explanation": explanation,
        "stability_pct": stab,
        "score_table": table,
    }


@router.get("/{experiment_id}/report")
def report(experiment_id: int, format: str = Query("html"), session: Session = Depends(get_session)):
    exp = Repository(session).get_experiment(experiment_id)
    if not exp or not exp.result_json:
        raise HTTPException(404, "results not found")
    result = BenchmarkResult.model_validate(exp.result_json)
    prefs = UserPreferences.model_validate(exp.preferences_json or {})
    df = results_to_dataframe(result)
    if format == "csv":
        return Response(export_csv(df), media_type="text/csv")
    if format == "json":
        return JSONResponse(result.model_dump())
    html = build_html_report(result, prefs)
    if format == "pdf":
        pdf = export_pdf(html)
        if pdf is None:
            raise HTTPException(501, "PDF extra not installed (pip install acbs[pdf])")
        return Response(pdf, media_type="application/pdf")
    return HTMLResponse(html)


@router.get("")
def list_experiments(session: Session = Depends(get_session)):
    return [
        {"id": e.id, "status": e.status, "dataset_id": e.dataset_id, "started_at": e.started_at}
        for e in Repository(session).list_experiments()
    ]


@router.delete("/{experiment_id}")
def delete_experiment(experiment_id: int, session: Session = Depends(get_session)):
    Repository(session).delete_experiment(experiment_id)
    return {"ok": True}
