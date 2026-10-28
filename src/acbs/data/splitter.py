from __future__ import annotations

import numpy as np
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)


def make_holdout_split(X, y, *, test_size: float = 0.2, seed: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def make_cv_splitter(
    *, cv_folds: int = 5, seed: int = 42
) -> StratifiedKFold | StratifiedShuffleSplit:
    """Build the outer-fold splitter shared by every contender.

    ``cv_folds >= 2`` gives stratified k-fold. A single fold (the protocol
    allows 1 to 10) falls back to one stratified holdout split, because
    ``StratifiedKFold`` rejects ``n_splits=1``.
    """
    if cv_folds <= 1:
        return StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    return StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)


def identical_folds(
    splitter: StratifiedKFold | StratifiedShuffleSplit, X, y
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Materialise the split list once so all frameworks see identical folds."""
    return list(splitter.split(X, y))
