"""Automatic target-column detection for uploaded CSVs.

The human's only job is uploading a file, so the system infers which column is
the target. Policy (preferred names, cardinality limits) lives in
``configs/target_detection.yaml``.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from acbs.exceptions import DatasetError
from acbs.utils.paths import load_yaml, repo_root

_POLICY_FILE = "target_detection.yaml"


def load_policy() -> dict:
    """Load the detection policy from ``configs/target_detection.yaml``."""
    return load_yaml(repo_root() / "configs" / _POLICY_FILE)


def _is_target_candidate(series: pd.Series, n_rows: int, policy: dict) -> bool:
    """True when a column could plausibly be a classification target."""
    if series.dropna().empty:
        return False
    nunique = int(series.nunique(dropna=True))
    if nunique < int(policy.get("min_classes", 2)):
        return False
    if nunique > int(policy.get("max_classes", 100)):
        return False
    ratio = nunique / max(n_rows, 1)
    return ratio <= float(policy.get("max_unique_ratio", 0.5))


def detect_target(df: pd.DataFrame, policy: dict | None = None) -> str:
    """Infer the target column of a dataframe.

    Raises ``DatasetError`` when no column looks like a classification target.
    """
    policy = policy if policy is not None else load_policy()
    if df.empty or df.shape[1] == 0:
        raise DatasetError("Empty dataset: no columns to choose a target from.")

    n_rows = len(df)
    lowered = {str(col).strip().lower(): col for col in df.columns}

    for name in policy.get("preferred_names") or []:
        col = lowered.get(str(name).strip().lower())
        if col is not None and _is_target_candidate(df[col], n_rows, policy):
            return str(col)

    # Convention fallback: the target is usually the last column.
    for col in reversed(list(df.columns)):
        if _is_target_candidate(df[col], n_rows, policy):
            return str(col)

    raise DatasetError(
        "Could not auto-detect a target column: no column has a plausible "
        "number of classes. Re-upload with an explicit target."
    )


def detect_target_in_file(path: str | Path, policy: dict | None = None) -> str:
    """Sniff a CSV/TSV file and return its inferred target column."""
    policy = policy if policy is not None else load_policy()
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"File not found: {path}")
    sep = "," if path.suffix.lower() == ".csv" else "\t"
    max_rows = int(policy.get("detection_max_rows", 200_000))
    try:
        df = pd.read_csv(path, sep=sep, nrows=max_rows)
    except Exception as exc:  # noqa: BLE001
        raise DatasetError(f"Could not read file for target detection: {exc}") from exc
    return detect_target(df, policy)


def detect_target_in_bytes(data: bytes, *, suffix: str = ".csv", policy: dict | None = None) -> str:
    """Sniff raw upload bytes for the target column without touching disk."""
    policy = policy if policy is not None else load_policy()
    sep = "," if suffix.lower() in {".csv", ""} else "\t"
    max_rows = int(policy.get("detection_max_rows", 200_000))
    try:
        df = pd.read_csv(io.BytesIO(data), sep=sep, nrows=max_rows)
    except Exception as exc:  # noqa: BLE001
        raise DatasetError(f"Could not read file for target detection: {exc}") from exc
    return detect_target(df, policy)
