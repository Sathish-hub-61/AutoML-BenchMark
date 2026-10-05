from __future__ import annotations

import pandas as pd
import pytest

from acbs.data.loader import LoadedDataset, _clean_frame, _validate, load_dataset
from acbs.exceptions import (
    AllNullColumnError,
    DuplicateColumnError,
    InsufficientRowsError,
    RegressionLikeTargetError,
    SingleClassError,
)


def test_clean_frame_duplicate_columns():
    df = pd.DataFrame([[1, 2]], columns=["col", "col"])
    with pytest.raises(DuplicateColumnError):
        _clean_frame(df)


def test_validate_insufficient_rows():
    df = pd.DataFrame({"a": range(10)})
    y = pd.Series([0, 1] * 5)
    with pytest.raises(InsufficientRowsError):
        _validate(df, y, "target")


def test_validate_single_class():
    df = pd.DataFrame({"a": range(60)})
    y = pd.Series([1] * 60)
    with pytest.raises(SingleClassError):
        _validate(df, y, "target")


def test_validate_all_null_column():
    df = pd.DataFrame({"a": range(60), "b": [None] * 60})
    y = pd.Series([0, 1] * 30)
    with pytest.raises(AllNullColumnError):
        _validate(df, y, "target")


def test_validate_regression_like_target():
    df = pd.DataFrame({"a": range(100)})
    y = pd.Series([float(i) for i in range(100)])
    with pytest.raises(RegressionLikeTargetError):
        _validate(df, y, "target")


def test_load_breast_cancer_dataset():
    ds = load_dataset("breast_cancer")
    assert isinstance(ds, LoadedDataset)
    assert ds.X.shape[0] >= 50
    assert ds.y.nunique() == 2
    assert "classes" in ds.meta
