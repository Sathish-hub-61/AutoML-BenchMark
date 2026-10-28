"""Tests for automatic target-column detection (acbs.data.target_detection)."""

from __future__ import annotations

import pandas as pd
import pytest

from acbs.data.target_detection import detect_target, detect_target_in_file
from acbs.exceptions import DatasetError

POLICY = {
    "preferred_names": ["target", "label", "class", "y"],
    "min_classes": 2,
    "max_classes": 100,
    "max_unique_ratio": 0.5,
    "detection_max_rows": 1000,
}


def _frame(n: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": range(n),
            "a": [i % 7 for i in range(n)],
            "b": [i * 0.5 for i in range(n)],
        }
    )


def test_prefers_named_target_column():
    df = _frame()
    df["label"] = ["yes" if i % 2 else "no" for i in range(len(df))]
    assert detect_target(df, POLICY) == "label"


def test_preferred_name_is_case_insensitive():
    df = _frame()
    df["TARGET"] = ["yes" if i % 2 else "no" for i in range(len(df))]
    assert detect_target(df, POLICY) == "TARGET"


def test_falls_back_to_last_valid_column():
    df = _frame()
    df["outcome"] = None  # exists but is all-null -> not a candidate
    df["churn"] = ["yes" if i % 3 else "no" for i in range(len(df))]
    # 'churn' is not a preferred name; 'b' is continuous; id is unique
    df = df.drop(columns=["outcome"])
    # target-last convention: churn is the last column and is valid
    assert detect_target(df, POLICY) == "churn"


def test_rejects_unique_id_column():
    df = _frame()  # id is unique, b is continuous, a has 7 classes
    df["id"] = range(len(df))
    # only 'a' qualifies (7 classes, ratio < 0.5)
    assert detect_target(df, POLICY) == "a"


def test_raises_when_nothing_looks_like_a_target():
    df = pd.DataFrame({"x": range(60), "y_float": [i * 0.123456 for i in range(60)]})
    with pytest.raises(DatasetError):
        detect_target(df, POLICY)


def test_detects_target_in_file(tmp_path):
    path = tmp_path / "sample.csv"
    rows = ["a,b,label"]
    for i in range(60):
        rows.append(f"{i % 5},{i * 1.5},{'yes' if i % 2 else 'no'}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    assert detect_target_in_file(path, POLICY) == "label"
