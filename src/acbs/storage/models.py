from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.dialects.sqlite import JSON as SQLITE_JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from sqlalchemy.types import JSON

from acbs.utils.paths import settings
from acbs.utils.time import utc_now


class Base(DeclarativeBase):
    pass


JSONType = JSON().with_variant(SQLITE_JSON(), "sqlite")



class DatasetRow(Base):
    __tablename__ = "datasets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255))
    source: Mapped[str] = mapped_column(String(64))
    n_rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    n_features: Mapped[int | None] = mapped_column(Integer, nullable=True)
    profile_json: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    file_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    experiments: Mapped[list[ExperimentRow]] = relationship(back_populates="dataset")



class ExperimentRow(Base):
    __tablename__ = "experiments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    config_json: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    preferences_json: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="queued")
    progress_json: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    lib_versions_json: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    result_json: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    dataset: Mapped[DatasetRow] = relationship(back_populates="experiments")
    model_runs: Mapped[list[ModelRunRow]] = relationship(back_populates="experiment", cascade="all, delete-orphan")
    recommendation: Mapped[RecommendationRow | None] = relationship(
        back_populates="experiment", uselist=False, cascade="all, delete-orphan"
    )


class ModelRunRow(Base):
    __tablename__ = "model_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"))
    model_name: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32))
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    cv_metrics_json: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    test_metrics_json: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    training_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    prediction_time_per_1k: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_params_json: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    fold_scores_json: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    experiment: Mapped[ExperimentRow] = relationship(back_populates="model_runs")


class RecommendationRow(Base):
    __tablename__ = "recommendations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"), unique=True)
    model_name: Mapped[str] = mapped_column(String(128))
    score_table_json: Mapped[dict | list | None] = mapped_column(JSONType, nullable=True)
    explanation_json: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    stability_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    experiment: Mapped[ExperimentRow] = relationship(back_populates="recommendation")


def make_engine(url: str | None = None):
    url = url or settings.database_url
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, future=True, connect_args=connect_args)


def make_session_factory(engine=None):
    engine = engine or make_engine()
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db(engine=None) -> None:
    engine = engine or make_engine()
    from alembic import command
    from alembic.config import Config

    from acbs.utils.paths import repo_root

    ini_path = repo_root() / "alembic.ini"
    alembic_cfg = Config(str(ini_path))
    alembic_cfg.set_main_option("script_location", str(repo_root() / "alembic"))
    alembic_cfg.set_main_option("sqlalchemy.url", str(engine.url))

    with engine.connect() as connection:
        alembic_cfg.attributes["connection"] = connection
        command.upgrade(alembic_cfg, "head")

