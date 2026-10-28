from __future__ import annotations

from fastapi import APIRouter

from acbs.api.schemas import FrameworkInfoResponse
from acbs.automl.availability import framework_availability

router = APIRouter(prefix="/frameworks", tags=["frameworks"])


@router.get("", response_model=list[FrameworkInfoResponse])
def list_frameworks() -> list[FrameworkInfoResponse]:
    """Return available AutoML frameworks with live environment install status."""
    return [
        FrameworkInfoResponse(
            id=fw["id"],
            name=fw["name"],
            strategy=fw["strategy"],
            installed=fw["installed"],
            version=fw["version"],
            os_supported=fw["os_supported"],
            reason=fw["reason"],
            default_budget_s=60,
        )
        for fw in framework_availability()
    ]


@router.post("/estimate", tags=["frameworks"])
def cost_estimate(payload: dict) -> dict:
    """Rough wall-clock estimate for a given experiment config.

    Returns ``total_wall_seconds`` and a breakdown so the UI can
    display a human-readable forecast before the user clicks Run.

    ``payload`` keys (all optional):
      - ``automl_frameworks``: list[str]  — selected framework IDs
      - ``automl_budget_s``: int  — per-framework budget
      - ``models``: list[str]  — classic models (used to estimate classic time)
      - ``n_datasets``: int  — default 1
    """
    fws = payload.get("automl_frameworks") or []
    budget_s: float = float(payload.get("automl_budget_s") or 60)
    n_datasets: int = int(payload.get("n_datasets") or 1)
    models = payload.get("models") or []
    n_models: int = len(models) if models and models != ["all"] else 8

    # Classic-tier estimate: ~15 s per model (rough; tuning triples it)
    classic_s = n_models * 15 * n_datasets
    # AutoML: budget + 25 % grace per framework
    automl_s = len(fws) * budget_s * 1.25 * n_datasets
    total_s = classic_s + automl_s

    return {
        "classic_estimate_s": round(classic_s),
        "automl_estimate_s": round(automl_s),
        "total_wall_seconds": round(total_s),
        "n_frameworks": len(fws),
        "budget_per_framework_s": budget_s,
        "n_datasets": n_datasets,
    }
