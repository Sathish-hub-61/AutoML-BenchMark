from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from acbs.benchmark.engine import BenchmarkConfig
from acbs.recommend.scoring import UserPreferences


class DatasetUploadResponse(BaseModel):
    id: int
    name: str
    profile: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
    # Target column actually used — detected automatically when the client
    # does not supply one, so a human never has to name it.
    target: str | None = None
    target_source: str | None = None  # "explicit" | "auto" | "registry"


class ExperimentCreate(BaseModel):
    dataset_id: int
    target: str | None = None
    config: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    automl: dict[str, Any] | None = None


class ExperimentStatus(BaseModel):
    id: int
    status: str
    progress: dict[str, Any] | None = None
    error: str | None = None
    # model/framework id -> {status, elapsed_s?, budget_s?}
    per_framework_status: dict[str, dict[str, Any]] = Field(default_factory=dict)
    elapsed_time_s: float | None = None


class RerankRequest(BaseModel):
    preferences: UserPreferences


class AdvisorQuestion(BaseModel):
    question: str | None = None


class FrameworkInfoResponse(BaseModel):
    id: str
    name: str
    strategy: str
    installed: bool
    version: str | None = None
    os_supported: bool = True
    reason: str | None = None
    default_budget_s: int = 60


class CostEstimateRequest(BaseModel):
    config: dict[str, Any] | BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    n_datasets: int = 1


class CostEstimateResponse(BaseModel):
    classic_seconds: float
    automl_seconds: float
    total_seconds: float
    total_minutes: float
