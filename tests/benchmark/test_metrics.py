from __future__ import annotations

import numpy as np

from acbs.benchmark.metrics import compute_metrics, get_scorers


def test_compute_metrics_binary():
    y_true = [0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 0, 1]
    y_proba = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.6, 0.4], [0.9, 0.1], [0.2, 0.8]])

    metrics = compute_metrics(y_true, y_pred, y_proba, classes=[0, 1])
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "roc_auc" in metrics
    assert metrics["accuracy"] > 0.5


def test_compute_metrics_multiclass():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 1]
    y_proba = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.7, 0.2, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.6, 0.3],
    ])

    metrics = compute_metrics(y_true, y_pred, y_proba, classes=[0, 1, 2])
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "roc_auc" in metrics


def test_scorers():
    scorers = get_scorers()
    assert "accuracy" in scorers
    assert "f1_macro" in scorers
