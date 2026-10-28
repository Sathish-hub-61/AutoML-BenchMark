from __future__ import annotations

import numpy as np
import pandas as pd

from acbs.data.splitter import identical_folds, make_cv_splitter


def _toy_data(n: int = 120) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series([0, 1] * (n // 2))
    return X, y


def test_single_fold_yields_one_stratified_split():
    X, y = _toy_data()
    cv = make_cv_splitter(cv_folds=1, seed=42)
    splits = identical_folds(cv, X, y)
    assert len(splits) == 1
    train_idx, test_idx = splits[0]
    assert set(train_idx).isdisjoint(set(test_idx))
    assert len(train_idx) + len(test_idx) == len(X)
    # Stratification: both classes on both sides of the split.
    assert set(y.iloc[train_idx]) == {0, 1}
    assert set(y.iloc[test_idx]) == {0, 1}


def test_single_fold_is_deterministic_for_a_seed():
    X, y = _toy_data()
    first = identical_folds(make_cv_splitter(cv_folds=1, seed=42), X, y)
    second = identical_folds(make_cv_splitter(cv_folds=1, seed=42), X, y)
    assert np.array_equal(first[0][0], second[0][0])
    assert np.array_equal(first[0][1], second[0][1])


def test_k_folds_unchanged():
    X, y = _toy_data()
    splits = identical_folds(make_cv_splitter(cv_folds=3, seed=42), X, y)
    assert len(splits) == 3
