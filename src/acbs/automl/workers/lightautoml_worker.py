#!/usr/bin/env python
"""LightAutoML worker — reads TrainSpec JSON from stdin, writes RunResult JSON to stdout.

Runs inside .envs/lightautoml/ virtualenv. No acbs imports.
LightAutoML uses a Bayesian optimisation approach over GBM + linear models.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd


def _aligned_proba(data, task_type: str) -> list:
    """Return probabilities shaped (n_samples, n_classes).

    LightAutoML returns a single column (probability of the positive class)
    for binary tasks and a (n, n_classes) matrix for multiclass; the central
    scorer contract requires one column per class in ``classes`` order.
    """
    arr = np.asarray(data, dtype=float)
    if task_type == "binary" and arr.ndim == 2 and arr.shape[1] == 1:
        arr = np.column_stack([1.0 - arr[:, 0], arr[:, 0]])
    return arr.tolist()


def main() -> None:
    spec = json.loads(sys.stdin.read())
    out: dict = {
        "status": "failed",
        "proba": None,
        "fit_time_s": 0.0,
        "predict_time_s": 0.0,
        "wall_time_s": 0.0,
        "framework_version": None,
        "best_pipeline": None,
        "error": None,
        "extra": {},
    }
    t0 = time.perf_counter()

    try:
        import lightautoml as _lama_mod
        from lightautoml.automl.presets.tabular_presets import TabularAutoML
        from lightautoml.tasks import Task

        out["framework_version"] = _lama_mod.__version__

        task_type = spec["task_type"]  # "binary" | "multiclass"
        lama_task = "binary" if task_type == "binary" else "multiclass"

        X_train = pd.DataFrame(spec["X_train"], columns=spec["feature_names"])
        y_train = pd.Series(spec["y_train"], name="target")
        X_test = pd.DataFrame(spec["X_test"], columns=spec["feature_names"])

        df_train = X_train.copy()
        if lama_task == "binary":
            df_train["target"] = y_train.astype(float)
        else:
            # LightAutoML detects multiclass from string labels
            df_train["target"] = y_train.astype(str)

        t_fit = time.perf_counter()
        automl = TabularAutoML(
            task=Task(lama_task),
            timeout=spec["time_budget_s"],
            cpu_limit=max(1, 4),
        )
        automl.fit_predict(
            df_train,
            roles={"target": "target"},
            verbose=0,
        )
        out["fit_time_s"] = round(time.perf_counter() - t_fit, 3)

        # Predict on hold-out test set
        t_pred = time.perf_counter()
        test_preds = automl.predict(X_test)
        out["predict_time_s"] = round(time.perf_counter() - t_pred, 3)

        proba = _aligned_proba(test_preds.data, task_type)
        out["proba"] = proba
        out["best_pipeline"] = "LightAutoML TabularAutoML"
        out["status"] = "ok"

    except Exception as exc:  # noqa: BLE001
        import traceback

        out["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1000:]}"
        out["status"] = "failed"
    finally:
        out["wall_time_s"] = round(time.perf_counter() - t0, 3)
        print(json.dumps(out))


if __name__ == "__main__":
    main()
