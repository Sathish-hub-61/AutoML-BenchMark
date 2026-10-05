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


class ExperimentCreate(BaseModel):
    dataset_id: int
    target: str | None = None
    config: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    preferences: UserPreferences = Field(default_factory=UserPreferences)


class ExperimentStatus(BaseModel):
    id: int
    status: str
    progress: dict[str, Any] | None = None
    error: str | None = None


class RerankRequest(BaseModel):
    preferences: UserPreferences
