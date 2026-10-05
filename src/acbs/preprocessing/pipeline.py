from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from acbs.data.profiler import DatasetProfile

TREE_FAMILIES = {"tree", "ensemble", "boosting"}


def _numeric_pipe(needs_scaling: bool, outlier_clipping: bool) -> SkPipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))
    return SkPipeline(steps)


def _categorical_pipe(cardinality: int, threshold: int) -> SkPipeline:
    imputer = SimpleImputer(strategy="most_frequent")
    if cardinality <= threshold:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=None)
    else:
        try:
            from sklearn.preprocessing import TargetEncoder

            encoder = TargetEncoder(random_state=42)
        except Exception:
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    return SkPipeline([("imputer", imputer), ("encoder", encoder)])


def build_preprocessor(
    profile: DatasetProfile,
    config: dict,
    model_family: str,
    X: pd.DataFrame | None = None,
) -> ColumnTransformer:
    drop = set(profile.suggested_drop_columns)
    numeric = [c for c in profile.numeric_columns if c not in drop]
    categorical = [c for c in profile.categorical_columns if c not in drop]
    if X is not None:
        numeric = [c for c in numeric if c in X.columns]
        categorical = [c for c in categorical if c in X.columns]
    needs_scaling = model_family not in TREE_FAMILIES
    high_card_thresh = int(config.get("high_cardinality", 15))
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                _numeric_pipe(needs_scaling, bool(config.get("outlier_clipping"))),
                numeric,
            )
        )
    cat_pipes = []
    if X is not None:
        for col in categorical:
            card = int(X[col].nunique(dropna=True))
            cat_pipes.append((f"cat_{col}", _categorical_pipe(card, high_card_thresh), [col]))
    else:
        if categorical:
            transformers.append(
                ("cat", _categorical_pipe(high_card_thresh, high_card_thresh), categorical)
            )
    transformers.extend(cat_pipes)
    pre = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    fs = config.get("feature_selection", "none")
    if fs == "variance":
        return SkPipeline([("pre", pre), ("vt", VarianceThreshold())])  # type: ignore[return-value]
    if fs == "kbest":
        return SkPipeline(  # type: ignore[return-value]
            [("pre", pre), ("kbest", SelectKBest(f_classif, k=int(config.get("k_best", 20))))]
        )
    return pre


def build_full_pipeline(model, preprocessor, imbalance_strategy: str):
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.under_sampling import RandomUnderSampler

    steps: list[tuple[str, Any]] = [("pre", preprocessor)]
    if imbalance_strategy == "smote":
        steps.append(("sampler", SMOTE(random_state=42)))
    elif imbalance_strategy == "random_under":
        steps.append(("sampler", RandomUnderSampler(random_state=42)))
    steps.append(("clf", model))
    return ImbPipeline(steps)
