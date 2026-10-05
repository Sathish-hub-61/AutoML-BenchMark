from __future__ import annotations

import numpy as np
import pandas as pd

from acbs.data.profiler import DatasetProfile, profile_dataset, profile_to_dict


def test_profile_dataset_synthetic():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "id": [f"ID_{i}" for i in range(n)],
            "num1": np.random.randn(n),
            "num2": np.random.exponential(size=n),
            "cat1": np.random.choice(["A", "B", "C"], size=n),
            "constant": [1] * n,
            "missing_col": [None if i % 2 == 0 else 1.0 for i in range(n)],
        }
    )
    y = pd.Series([0] * 80 + [1] * 20)

    profile = profile_dataset(df, y, imbalance_threshold=3.0)
    assert isinstance(profile, DatasetProfile)
    assert profile.n_rows == 100
    assert profile.n_features == 6
    assert profile.is_imbalanced is True
    assert profile.imbalance_ratio == 4.0
    assert "constant" in profile.suggested_drop_columns
    assert "id" in profile.suggested_drop_columns
    assert len(profile.warnings) > 0

    pdict = profile_to_dict(profile)
    assert isinstance(pdict, dict)
    assert pdict["n_rows"] == 100
