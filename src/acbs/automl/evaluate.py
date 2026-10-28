"""Evaluate AutoML predictions centrally using acbs benchmark metrics."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from acbs.automl.contracts import RunResult

logger = logging.getLogger(__name__)


def evaluate_automl_result(
    result: RunResult,
    y_test: pd.Series | np.ndarray | list[int | str | float],
    classes: list[Any],
) -> dict[str, Any]:
    """Compute central metrics on AutoML predictions and verify class alignment.

    Parameters
    ----------
    result:
        RunResult returned by run_framework.
    y_test:
        True test labels.
    classes:
        Ordered list of unique class labels expected from the task.

    Returns
    -------
    dict with test_metrics, y_pred, y_true, y_proba, and error.
    """
    y_true_arr = np.asarray(y_test)
    n_samples = len(y_true_arr)
    n_classes = len(classes)

    if result.status != "ok" or result.proba is None:
        return {
            "test_metrics": {},
            "y_pred": [],
            "y_true": [
                int(x) if isinstance(x, (int, np.integer)) else x for x in y_true_arr.tolist()
            ],
            "y_proba": None,
            "error": result.error,
        }

    proba = np.asarray(result.proba, dtype=float)

    # Validate shape
    if proba.ndim == 1:
        if n_classes == 2:
            proba = np.column_stack([1.0 - proba, proba])
        else:
            raise ValueError(
                f"1D probabilities returned for multiclass task with {n_classes} classes; shape={proba.shape}"
            )

    if proba.shape[0] != n_samples:
        raise ValueError(
            f"Prediction sample count mismatch: expected {n_samples}, got {proba.shape[0]}"
        )

    if proba.shape[1] != n_classes:
        raise ValueError(
            f"Probability column count mismatch: expected {n_classes} ({classes}), got {proba.shape[1]}"
        )

    # Validate probability normalization (~1.0 per row)
    row_sums = proba.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=0.05):
        logger.warning(
            "[%s] probabilities do not sum to 1.0 (min=%.3f, max=%.3f); normalizing",
            result.framework_id,
            float(row_sums.min()),
            float(row_sums.max()),
        )
        row_sums_safe = np.where(row_sums == 0, 1.0, row_sums)[:, None]
        proba = proba / row_sums_safe

    # Compute y_pred by argmax over classes
    best_indices = np.argmax(proba, axis=1)
    class_arr = np.asarray(classes)
    y_pred_arr = class_arr[best_indices]

    # Compute all standard acbs metrics centrally
    # Imported lazily: importing acbs.benchmark at module load pulls in the
    # engine, which imports this module — a cycle that breaks any entry point
    # landing on evaluate first (e.g. running tests/automl/ in isolation).
    from acbs.benchmark.metrics import compute_metrics

    test_metrics = compute_metrics(y_true_arr, y_pred_arr, proba, classes)

    return {
        "test_metrics": test_metrics,
        "y_pred": y_pred_arr.tolist(),
        "y_true": y_true_arr.tolist(),
        "y_proba": proba.tolist(),
        "error": None,
    }
