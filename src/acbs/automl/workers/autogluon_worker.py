#!/usr/bin/env python
"""AutoGluon worker — reads TrainSpec JSON from stdin, writes RunResult JSON to stdout.

Runs inside .envs/autogluon/ virtualenv. No acbs imports.
"""

from __future__ import annotations

import json
import sys
import tempfile
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
        import autogluon.tabular
        from autogluon.tabular import TabularPredictor

        out["framework_version"] = autogluon.tabular.__version__

        task_type = spec["task_type"]  # "binary" | "multiclass"
        ag_problem = "binary" if task_type == "binary" else "multiclass"

        X_train = pd.DataFrame(spec["X_train"], columns=spec["feature_names"])
        y_train = pd.Series(spec["y_train"], name="target")
        X_test = pd.DataFrame(spec["X_test"], columns=spec["feature_names"])

        df_train = X_train.copy()
        df_train["target"] = y_train

        # AutoGluon saves models; use a temp dir that we'll clean up
        with tempfile.TemporaryDirectory(prefix="acbs_ag_") as tmp:
            t_fit = time.perf_counter()
            predictor = TabularPredictor(
                label="target",
                problem_type=ag_problem,
                path=tmp,
                verbosity=0,
            ).fit(
                df_train,
                time_limit=spec["time_budget_s"],
                presets="medium_quality",
            )
            out["fit_time_s"] = round(time.perf_counter() - t_fit, 3)

            t_pred = time.perf_counter()
            proba_df = predictor.predict_proba(X_test)
            out["predict_time_s"] = round(time.perf_counter() - t_pred, 3)

        # proba_df columns may be class labels; convert to ordered array
        proba = proba_df.values  # shape (n_test, n_classes)
        out["proba"] = proba.tolist()
        # Metadata only: an inspection-API mismatch across AutoGluon versions
        # must never fail a model that already trained and predicted fine.
        try:
            out["best_pipeline"] = str(predictor.get_model_best())
        except Exception:  # noqa: BLE001
            out["best_pipeline"] = str(getattr(predictor, "model_best", "") or "")
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
