from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


def repo_root() -> Path:
    env = os.getenv("ACBS_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "configs" / "default.yaml").exists():
            return parent
    return Path.cwd()


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_default_config() -> dict:
    return load_yaml(repo_root() / "configs" / "default.yaml")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./acbs.db"
    max_upload_mb: int = 50


settings = Settings()
