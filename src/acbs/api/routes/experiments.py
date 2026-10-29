from __future__ import annotations

import threading
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from sqlalchemy.orm import Session

from acbs.api.deps import SessionLocal, get_session
from acbs.api.schemas import AdvisorQuestion, ExperimentCreate, ExperimentStatus, RerankRequest
from acbs.automl.selection import select_frameworks
from acbs.benchmark.aggregate import results_to_dataframe
from acbs.benchmark.engine import BenchmarkResult, run_benchmark
from acbs.data.loader import load_dataset_row
from acbs.recommend import advisor
from acbs.recommend.explain import generate_explanation
from acbs.recommend.scoring import UserPreferences, score_models, sensitivity_analysis
from acbs.reporting.report import build_html_report, export_csv, export_pdf
from acbs.storage.repository import Repository
from acbs.utils.jsonsafe import json_safe
from acbs.utils.logging import get_logger
from acbs.utils.time import utc_now

logger = get_logger(__name__)
router = APIRouter(prefix="/experiments", tags=["experiments"])

# Experiments cancelled while queued/running. Kept in-process: a restart clears it,
# which is acceptable because the worker itself dies with the process.
_CANCELLED: set[int] = set()
_CANCEL_LOCK = threading.Lock()


def _request_cancel(experiment_id: int) -> None:
    with _CANCEL_LOCK:
        _CANCELLED.add(experiment_id)


def _is_cancelled(experiment_id: int) -> bool:
    with _CANCEL_LOCK:
        return experiment_id in _CANCELLED


def _forget_cancel(experiment_id: int) -> None:
    """Drop the flag once the run is over so a recycled id cannot be cancelled."""
    with _CANCEL_LOCK:
        _CANCELLED.discard(experiment_id)


def _elapsed_seconds(started_at, finished_at) -> float | None:
    """Seconds between two datetimes, tolerating naive/aware mixes from SQLite."""
    if started_at is None:
        return None
    end = finished_at or utc_now()
    try:
        return (end - started_at).total_seconds()
    except TypeError:
        return (end.replace(tzinfo=None) - started_at.replace(tzinfo=None)).total_seconds()


def _recommend(result: BenchmarkResult, prefs: UserPreferences):
    df = results_to_dataframe(result)
    scored = score_models(df, prefs)
    eligible = scored[~scored["excluded"]]
    if eligible.empty:
        raise ValueError("No models satisfied the constraints")
    rec = eligible.sort_values("rank").iloc[0]
    stability = sensitivity_analysis(df, prefs)
    explanation = generate_explanation(rec, df, result.profile, prefs, stability)
    table = json_safe(
        scored.drop(columns=["best_params"], errors="ignore").to_dict(orient="records")
    )
    return rec, table, json_safe(explanation), stability.get(rec["model_name"], 0.0) * 100, scored


def _execute(experiment_id: int) -> None:
    session = SessionLocal()
    repo = Repository(session)
    try:
        exp = repo.get_experiment(experiment_id)
        if exp is None:
            return
        if _is_cancelled(experiment_id):
            repo.update_status(
                experiment_id,
                "cancelled",
                started_at=utc_now(),
                finished_at=utc_now(),
                error="cancelled before start",
            )
            return
        repo.update_status(experiment_id, "running", started_at=utc_now(), error=None)
        dataset = load_dataset_row(exp.dataset)
        cfg = dict(exp.config_json or {})
        automl_ids = set(cfg.get("automl_frameworks") or [])
        automl_budget = cfg.get("automl_budget_s")
        fw_status: dict[str, dict[str, Any]] = {}

        def progress(model_name, step, total, message, status=None, elapsed_s=None):
            pct = int(100 * step / max(total, 1))
            if status:
                entry = fw_status.setdefault(model_name, {})
                entry["status"] = status
                if elapsed_s is not None:
                    entry["elapsed_s"] = elapsed_s
                if model_name in automl_ids and automl_budget:
                    entry["budget_s"] = int(automl_budget)
            s = SessionLocal()
            try:
                Repository(s).update_status(
                    experiment_id,
                    "running",
                    progress_json={
                        "pct": pct,
                        "current_model": model_name,
                        "message": message,
                        "per_framework_status": {k: dict(v) for k, v in fw_status.items()},
                    },
                )
            finally:
                s.close()

        result = run_benchmark(
            dataset,
            cfg,
            progress_callback=progress,
            should_stop=lambda: _is_cancelled(experiment_id),
        )
        repo.save_results(experiment_id, result.model_dump())
        final_progress = {
            "pct": 100,
            "current_model": None,
            "message": "done",
            "per_framework_status": {k: dict(v) for k, v in fw_status.items()},
        }

        if _is_cancelled(experiment_id):
            final_progress["message"] = "cancelled by user"
            repo.update_status(
                experiment_id,
                "cancelled",
                finished_at=utc_now(),
                error="cancelled by user; partial results stored",
                progress_json=final_progress,
            )
            return

        prefs = UserPreferences.model_validate(exp.preferences_json or {})
        try:
            rec, table, explanation, stab, _ = _recommend(result, prefs)
        except Exception as exc:
            # The benchmark itself succeeded: never hide saved results behind a
            # recommendation/constraint failure.
            logger.exception("recommendation failed for experiment %s", experiment_id)
            final_progress["message"] = "finished; recommendation unavailable"
            repo.update_status(
                experiment_id,
                "completed",
                finished_at=utc_now(),
                error=f"recommendation failed: {exc}",
                progress_json=final_progress,
            )
            return
        repo.save_recommendation(experiment_id, rec["model_name"], table, explanation, stab)
        final_progress["current_model"] = rec["model_name"]
        repo.update_status(
            experiment_id,
            "completed",
            finished_at=utc_now(),
            error=None,
            progress_json=final_progress,
        )
    except Exception as exc:
        logger.exception("experiment %s failed", experiment_id)
        repo.update_status(experiment_id, "failed", error=str(exc), finished_at=utc_now())

    finally:
        _forget_cancel(experiment_id)
        session.close()


@router.post("", status_code=202)
def create_experiment(
    payload: ExperimentCreate, background: BackgroundTasks, session: Session = Depends(get_session)
):
    repo = Repository(session)
    ds = repo.get_dataset(payload.dataset_id)
    if not ds:
        raise HTTPException(404, "dataset not found")
    config = payload.config.model_dump()
    # The UI sends the AutoML selection as a separate `automl` block; fold it into
    # the stored config so the engine actually runs those frameworks.
    automl = payload.automl or {}
    if automl.get("auto") or automl.get("frameworks") == ["auto"]:
        # Zero-interaction mode: the system picks the frameworks itself from
        # installed, OS-supported environments, sized to the dataset.
        selection = select_frameworks(
            ds.profile_json,
            budget_s=automl.get("budget_s"),
        )
        config["automl_frameworks"] = selection.frameworks
        config["automl_budget_s"] = selection.budget_s
        config["automl_auto"] = True
        config["automl_selection"] = selection.as_dict()
        if automl.get("n_jobs"):
            config["automl_n_jobs"] = int(automl["n_jobs"])
    else:
        if automl.get("frameworks"):
            config["automl_frameworks"] = list(automl["frameworks"])
        if automl.get("budget_s"):
            config["automl_budget_s"] = int(automl["budget_s"])
        if automl.get("n_jobs"):
            config["automl_n_jobs"] = int(automl["n_jobs"])
    exp = repo.create_experiment(
        payload.dataset_id,
        config,
        payload.preferences.model_dump(),
        payload.config.seed,
    )
    background.add_task(_execute, exp.id)
    return {
        "id": exp.id,
        "status": exp.status,
        "automl_frameworks": config.get("automl_frameworks") or [],
        "automl_budget_s": config.get("automl_budget_s"),
        "automl_auto": bool(config.get("automl_auto")),
        "automl_selection": config.get("automl_selection"),
    }


@router.get("/{experiment_id}/status", response_model=ExperimentStatus)
def status(experiment_id: int, session: Session = Depends(get_session)):
    exp = Repository(session).get_experiment(experiment_id)
    if not exp:
        raise HTTPException(404, "experiment not found")
    progress = exp.progress_json or {}
    return ExperimentStatus(
        id=exp.id,
        status=exp.status,
        progress=progress,
        error=exp.error,
        per_framework_status=progress.get("per_framework_status") or {},
        elapsed_time_s=_elapsed_seconds(exp.started_at, exp.finished_at),
    )


@router.get("/{experiment_id}/results")
def results(experiment_id: int, session: Session = Depends(get_session)):
    exp = Repository(session).get_experiment(experiment_id)
    if not exp:
        raise HTTPException(404, "experiment not found")
    if exp.status != "completed":
        raise HTTPException(409, f"experiment status is {exp.status}")
    return json_safe(exp.result_json)


@router.get("/{experiment_id}/recommendation")
def recommendation(experiment_id: int, session: Session = Depends(get_session)):
    exp = Repository(session).get_experiment(experiment_id)
    if not exp or not exp.recommendation:
        raise HTTPException(404, "recommendation not found")
    rec = exp.recommendation
    return json_safe(
        {
            "model_name": rec.model_name,
            "explanation": rec.explanation_json,
            "stability_pct": rec.stability_pct,
            "score_table": rec.score_table_json,
        }
    )


@router.post("/{experiment_id}/advisor")
def advisor_explanation(
    experiment_id: int,
    payload: AdvisorQuestion | None = None,
    session: Session = Depends(get_session),
):
    """Let the Benchmark Advisor narrate the stored recommendation.

    The narrative is generated by the hosted Respan gateway from the stored
    explanation, so it can only restate those numbers. The advisor is optional:
    without ``RESPAN_API_KEY`` it answers 503 and the UI keeps the template text.
    """
    exp = Repository(session).get_experiment(experiment_id)
    if not exp or not exp.recommendation:
        raise HTTPException(404, "recommendation not found")
    if not advisor.is_available():
        raise HTTPException(503, "advisor unavailable: set RESPAN_API_KEY")
    question = (payload.question if payload else None) or None
    try:
        narrative = advisor.narrate(exp.recommendation.explanation_json or {}, question=question)
    except Exception as exc:
        # The advisor is decorative: a gateway failure must not hide the results.
        logger.exception("advisor request failed for experiment %s", experiment_id)
        raise HTTPException(502, f"advisor request failed: {exc}") from exc
    return {"narrative": narrative, "model": advisor.advisor_settings()["model"]}


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
    return json_safe(
        {
            "model_name": rec["model_name"],
            "explanation": explanation,
            "stability_pct": stab,
            "score_table": table,
        }
    )


@router.get("/{experiment_id}/report")
def report(
    experiment_id: int, format: str = Query("html"), session: Session = Depends(get_session)
):
    exp = Repository(session).get_experiment(experiment_id)
    if not exp or not exp.result_json:
        raise HTTPException(404, "results not found")
    result = BenchmarkResult.model_validate(exp.result_json)
    prefs = UserPreferences.model_validate(exp.preferences_json or {})
    df = results_to_dataframe(result)
    if format == "csv":
        return Response(export_csv(df), media_type="text/csv")
    if format == "json":
        return JSONResponse(json_safe(result.model_dump()))
    html = build_html_report(result, prefs)
    if format == "pdf":
        pdf = export_pdf(html)
        if pdf is None:
            raise HTTPException(501, "PDF extra not installed (pip install acbs[pdf])")
        return Response(pdf, media_type="application/pdf")
    return HTMLResponse(html)


@router.get("/{experiment_id}/export-model")
def export_model(experiment_id: int, session: Session = Depends(get_session)):
    repo = Repository(session)
    exp = repo.get_experiment(experiment_id)
    if not exp or not exp.recommendation:
        raise HTTPException(404, "Experiment or recommendation not found")
    try:
        from acbs.models.export import export_experiment_model

        zip_path = export_experiment_model(exp)
        return FileResponse(
            path=str(zip_path),
            filename=zip_path.name,
            media_type="application/zip",
        )
    except Exception as exc:
        logger.error(f"Failed to export model for experiment {experiment_id}: {exc}", exc_info=True)
        raise HTTPException(500, f"Model export failed: {exc}") from exc


@router.get("")
def list_experiments(session: Session = Depends(get_session)):
    return [
        {"id": e.id, "status": e.status, "dataset_id": e.dataset_id, "started_at": e.started_at}
        for e in Repository(session).list_experiments()
    ]


@router.delete("/{experiment_id}")
def delete_experiment(experiment_id: int, session: Session = Depends(get_session)):
    """Cancel a queued/running experiment; delete a finished one."""
    repo = Repository(session)
    exp = repo.get_experiment(experiment_id)
    if exp is None:
        return {"ok": True, "action": "absent"}
    if exp.status in {"queued", "running", "cancelling"}:
        _request_cancel(experiment_id)
        progress = dict(exp.progress_json or {})
        progress["message"] = "cancellation requested"
        repo.update_status(experiment_id, "cancelling", progress_json=progress)
        return {"ok": True, "action": "cancelling"}
    repo.delete_experiment(experiment_id)
    _forget_cancel(experiment_id)
    return {"ok": True, "action": "deleted"}
