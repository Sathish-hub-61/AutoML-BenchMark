#!/usr/bin/env python
"""FLAML worker — reads TrainSpec JSON from stdin, writes RunResult JSON to stdout.

Runs inside .envs/flaml/ virtualenv. No acbs imports.
"""

from __future__ import annotations

import json
import sys
import time

import pandas as pd


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
        import flaml
        from flaml import AutoML

        out["framework_version"] = flaml.__version__

        task_type = spec["task_type"]  # "binary" | "multiclass"
        flaml_task = "classification"  # flaml uses "classification" for both

        X_train = pd.DataFrame(spec["X_train"], columns=spec["feature_names"])
        y_train = pd.Series(spec["y_train"])
        X_test = pd.DataFrame(spec["X_test"], columns=spec["feature_names"])

        automl = AutoML()

        t_fit = time.perf_counter()
        automl.fit(
            X_train,
            y_train,
            task=flaml_task,
            time_budget=spec["time_budget_s"],
            seed=spec["seed"],
            metric="roc_auc" if task_type == "binary" else "macro_f1",
            verbose=0,
        )
        out["fit_time_s"] = round(time.perf_counter() - t_fit, 3)

        t_pred = time.perf_counter()
        proba = automl.predict_proba(X_test)
        out["predict_time_s"] = round(time.perf_counter() - t_pred, 3)

        out["proba"] = proba.tolist()
        out["best_pipeline"] = str(automl.best_estimator)
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
