from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_roc_auc_scorer(y_true, y_proba) -> float:
    """ROC-AUC scorer safe for both binary (2-class proba) and multiclass."""
    try:
        proba = np.asarray(y_proba)
        if proba.ndim == 1:
            return float(roc_auc_score(y_true, proba))
        if proba.shape[1] == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        return float(
            roc_auc_score(
                y_true, proba, multi_class="ovr", average="macro", labels=np.unique(y_true)
            )
        )
    except Exception:
        return float("nan")


def _safe_auc(y_true, y_proba, classes) -> float:
    try:
        n_classes = len(classes)
        if n_classes == 2:
            if y_proba.ndim == 2:
                scores = y_proba[:, 1]
            else:
                scores = y_proba
            return float(roc_auc_score(y_true, scores))
        return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=classes))
    except Exception:
        return float("nan")


def compute_metrics(y_true, y_pred, y_proba, classes) -> dict[str, float]:
    labels = list(classes)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": _safe_auc(y_true, y_proba, labels) if y_proba is not None else float("nan"),
    }


def get_scorers() -> dict[str, Any]:
    return {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
        "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
        "balanced_accuracy": "balanced_accuracy",
        "roc_auc_ovr": make_scorer(_safe_roc_auc_scorer, response_method="predict_proba"),
    }


def predict_proba_or_decision(model, X) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X))
        if scores.ndim == 1:
            exp = np.exp(np.clip(scores, -50, 50))
            p = exp / (1 + exp)
            return np.column_stack([1 - p, p])
        exp = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)
    return None
