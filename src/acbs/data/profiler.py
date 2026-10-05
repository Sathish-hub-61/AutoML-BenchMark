from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class DatasetProfile(BaseModel):
    n_rows: int
    n_features: int
    n_numeric: int
    n_categorical: int
    n_high_cardinality_categorical: int
    missing_ratio: float
    missing_top: dict[str, float]
    n_duplicate_rows: int
    class_distribution: dict[str, float]
    class_counts: dict[str, int]
    n_classes: int
    imbalance_ratio: float
    is_imbalanced: bool
    task_type: str
    skew_share: float
    outlier_share: float
    dimensionality_ratio: float
    estimated_memory_mb: float
    size_category: str
    warnings: list[str] = Field(default_factory=list)
    suggested_drop_columns: list[str] = Field(default_factory=list)
    numeric_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)


def _is_categorical(series: pd.Series) -> bool:
    return series.dtype == object or str(series.dtype) in {"category", "bool", "boolean"} or (
        pd.api.types.is_integer_dtype(series) and series.nunique(dropna=True) <= 12
    )


def profile_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    imbalance_threshold: float = 3.0,
    high_cardinality: int = 50,
    id_unique_ratio: float = 0.95,
) -> DatasetProfile:
    n_rows, n_features = X.shape
    numeric_cols: list[str] = []
    cat_cols: list[str] = []
    high_card = 0
    drop: list[str] = []
    warnings: list[str] = []

    for col in X.columns:
        s = X[col]
        nunique = int(s.nunique(dropna=True))
        if nunique <= 1:
            drop.append(str(col))
            warnings.append(f"Constant column: {col}")
            continue
        unique_ratio = nunique / max(n_rows, 1)
        if unique_ratio > id_unique_ratio and (
            pd.api.types.is_integer_dtype(s) or s.dtype == object
        ):
            drop.append(str(col))
            warnings.append(f"Possible ID column: {col}")
            continue
        if _is_categorical(s) and not pd.api.types.is_float_dtype(s):
            cat_cols.append(str(col))
            if nunique > high_cardinality:
                high_card += 1
        else:
            numeric_cols.append(str(col))

    missing_ratio = float(X.isna().mean().mean()) if n_features else 0.0
    missing_top = (
        X.isna().mean().sort_values(ascending=False).head(10).astype(float).to_dict()
        if n_features
        else {}
    )
    bad_missing = [c for c, r in missing_top.items() if r > 0.4]
    if bad_missing:
        warnings.append(f"{len(bad_missing)} columns have >40% missing values")

    counts = y.value_counts(dropna=False)
    class_counts = {str(k): int(v) for k, v in counts.items()}
    class_dist = {str(k): float(v / n_rows) for k, v in counts.items()}
    n_classes = int(y.nunique())
    imb = float(counts.max() / max(counts.min(), 1))
    is_imb = imb > imbalance_threshold
    if is_imb:
        warnings.append(f"Severe class imbalance 1:{imb:.1f}" if imb > 10 else f"Class imbalance {imb:.1f}:1")

    numeric = X[numeric_cols].select_dtypes(include=[np.number])
    skew_share = 0.0
    outlier_share = 0.0
    if numeric.shape[1]:
        skew = numeric.skew(numeric_only=True).abs()
        skew_share = float((skew > 1).mean()) if len(skew) else 0.0
        q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
        iqr = (q3 - q1).replace(0, np.nan)
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (numeric.lt(lower) | numeric.gt(upper))
        outlier_share = float(mask.stack().mean()) if mask.size else 0.0

    if n_rows < 5000:
        size = "small"
    elif n_rows <= 100_000:
        size = "medium"
    else:
        size = "large"

    return DatasetProfile(
        n_rows=n_rows,
        n_features=n_features,
        n_numeric=len(numeric_cols),
        n_categorical=len(cat_cols),
        n_high_cardinality_categorical=high_card,
        missing_ratio=missing_ratio,
        missing_top={str(k): float(v) for k, v in missing_top.items()},
        n_duplicate_rows=int(X.duplicated().sum()),
        class_distribution=class_dist,
        class_counts=class_counts,
        n_classes=n_classes,
        imbalance_ratio=imb,
        is_imbalanced=is_imb,
        task_type="binary" if n_classes == 2 else "multiclass",
        skew_share=skew_share,
        outlier_share=outlier_share,
        dimensionality_ratio=float(n_features / max(n_rows, 1)),
        estimated_memory_mb=float(X.memory_usage(deep=True).sum() / 1e6),
        size_category=size,
        warnings=warnings,
        suggested_drop_columns=drop,
        numeric_columns=numeric_cols,
        categorical_columns=cat_cols,
    )


def profile_to_dict(profile: DatasetProfile) -> dict[str, Any]:
    return profile.model_dump()
