from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.preprocessing import LabelEncoder

from acbs.exceptions import (
    AllNullColumnError,
    DatasetError,
    DuplicateColumnError,
    InsufficientRowsError,
    RegressionLikeTargetError,
    SingleClassError,
    TargetNotFoundError,
    UnsupportedSourceError,
)
from acbs.utils.paths import load_yaml, repo_root

MIN_ROWS = 50


@dataclass
class LoadedDataset:
    X: pd.DataFrame
    y: pd.Series
    meta: dict[str, Any] = field(default_factory=dict)


def load_registry() -> list[dict[str, Any]]:
    payload = load_yaml(repo_root() / "data" / "registry.yaml")
    return list(payload.get("datasets", []))


def get_registry_entry(name: str) -> dict[str, Any] | None:
    for row in load_registry():
        if row["name"] == name:
            return row
    return None


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        dups = df.columns[df.columns.duplicated()].tolist()
        raise DuplicateColumnError(f"Duplicate columns: {dups}")
    for col in df.columns:
        if df[col].dtype == object or isinstance(df[col].dtype, pd.StringDtype):
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                # Not numeric text; leave the column untouched.
                pass
    return df


def _validate(X: pd.DataFrame, y: pd.Series, target: str) -> None:
    if len(X) < MIN_ROWS:
        raise InsufficientRowsError(f"Need at least {MIN_ROWS} rows, got {len(X)}")
    if y.nunique(dropna=True) < 2:
        raise SingleClassError("Target must have at least 2 classes")
    null_cols = [c for c in X.columns if X[c].isna().all()]
    if null_cols:
        raise AllNullColumnError(f"All-null columns: {null_cols}")
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 50 and y.nunique() > 0.2 * len(y):
        raise RegressionLikeTargetError(
            f"Target '{target}' looks like regression ({y.nunique()} unique numeric values)"
        )


def _encode_target(y: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    encoder = LabelEncoder()
    values = encoder.fit_transform(y.astype(str))
    mapping = {str(cls): int(i) for i, cls in enumerate(encoder.classes_)}
    return pd.Series(values, name="target", index=y.index), {
        "label_mapping": mapping,
        "classes": [str(c) for c in encoder.classes_],
    }


def _from_frame(df: pd.DataFrame, target: str, meta: dict[str, Any]) -> LoadedDataset:
    df = _clean_frame(df)
    if target not in df.columns:
        raise TargetNotFoundError(
            f"Target column '{target}' not found. Columns: {list(df.columns)}"
        )
    y_raw = df[target]
    X = df.drop(columns=[target])
    _validate(X, y_raw, target)
    y, enc = _encode_target(y_raw)
    meta = {
        **meta,
        **enc,
        "target_column": target,
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
    }
    return LoadedDataset(X=X.reset_index(drop=True), y=y.reset_index(drop=True), meta=meta)


def _cache_path(name: str) -> Path:
    return repo_root() / "data" / "processed" / f"{name}.parquet"


def _sklearn_dataset(identifier: str) -> pd.DataFrame:
    loaders = {"breast_cancer": load_breast_cancer, "wine": load_wine}
    if identifier not in loaders:
        raise DatasetError(f"Unknown sklearn dataset '{identifier}'")
    bunch = loaders[identifier](as_frame=True)
    frame = bunch.frame.copy()
    if "target" not in frame.columns:
        frame["target"] = bunch.target
    return frame


def _openml_dataset(identifier: str | int, target: str | None) -> pd.DataFrame:
    import openml

    ds = openml.datasets.get_dataset(int(identifier))
    X, y, _, _ = ds.get_data(target=ds.default_target_attribute, dataset_format="dataframe")
    frame = X.copy()
    tcol = target or ds.default_target_attribute or "target"
    frame[tcol] = y
    return frame


def load_local(path: str | Path, target: str) -> LoadedDataset:
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        df = pd.read_csv(path, sep="," if suffix == ".csv" else "\t")
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise DatasetError(f"Unsupported file type: {suffix}")
    return _from_frame(df, target, {"source": "local", "path": str(path), "name": path.stem})


def load_dataset(name_or_path: str, target: str | None = None) -> LoadedDataset:
    """Load a registry dataset, local file, or cached parquet."""
    path = Path(name_or_path)
    if path.exists() and path.is_file():
        if target is None:
            raise TargetNotFoundError("target is required for local files")
        return load_local(path, target)

    entry = get_registry_entry(name_or_path)
    if entry is None:
        raise DatasetError(f"Unknown dataset '{name_or_path}'")
    target = target or entry["target_column"]
    cache = _cache_path(entry["name"])
    if cache.exists():
        df = pd.read_parquet(cache)
        return _from_frame(df, target, {**entry, "cached": True})

    source = entry["source"]
    if source == "sklearn":
        df = _sklearn_dataset(str(entry["identifier"]))
    elif source == "openml":
        df = _openml_dataset(entry["identifier"], target)
    elif source == "local":
        return load_local(entry["identifier"], target)
    elif source == "kaggle":
        raise UnsupportedSourceError(
            f"Dataset '{entry['name']}' is Kaggle-only. Download it manually into data/raw/."
        )
    else:
        raise UnsupportedSourceError(f"Unsupported source '{source}'")

    sample_rows = entry.get("sample_rows")
    if sample_rows and len(df) > int(sample_rows):
        df = df.sample(n=int(sample_rows), random_state=42)
    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(cache, index=False)
    except Exception:
        pass
    return _from_frame(df, target, {**entry, "cached": False})


def load_dataset_row(row: Any) -> LoadedDataset:
    """Load the dataset a stored ``DatasetRow`` refers to.

    Uploaded files are re-read from ``data/raw``; rows from other sources
    fall back to :func:`load_dataset`. The target column comes from the
    profile's ``meta.target_column`` recorded when the dataset was stored.
    """
    meta = (getattr(row, "profile_json", None) or {}).get("meta") or {}
    target = meta.get("target_column")
    if row.source == "upload":
        if not target:
            raise DatasetError(f"Uploaded dataset '{row.name}' has no recorded target column")
        return load_local(repo_root() / "data" / "raw" / row.name, target)
    return load_dataset(row.name, target)
