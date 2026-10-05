from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def make_holdout_split(X, y, *, test_size: float = 0.2, seed: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def make_cv_splitter(*, cv_folds: int = 5, seed: int = 42) -> StratifiedKFold:
    return StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)


def identical_folds(splitter: StratifiedKFold, X, y) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(splitter.split(X, y))
