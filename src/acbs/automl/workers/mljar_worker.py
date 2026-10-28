#!/usr/bin/env python
"""MLJAR worker — reads TrainSpec JSON from stdin, writes RunResult JSON to stdout.

Runs inside .envs/mljar/ virtualenv. No acbs imports.
MLJAR uses a custom data-science pipeline with stacked ensembling.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time

import numpy as np
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
        import supervised
        from supervised.automl import AutoML

        out["framework_version"] = supervised.__version__

        X_train = pd.DataFrame(spec["X_train"], columns=spec["feature_names"])
        y_train = pd.Series(spec["y_train"])
        X_test = pd.DataFrame(spec["X_test"], columns=spec["feature_names"])

        with tempfile.TemporaryDirectory(prefix="acbs_mljar_") as tmp:
            t_fit = time.perf_counter()
            automl = AutoML(
                results_path=tmp,
                mode="Explain",
                total_time_limit=spec["time_budget_s"],
                random_state=spec["seed"],
                verbose=0,
            )
            automl.fit(X_train, y_train)
            out["fit_time_s"] = round(time.perf_counter() - t_fit, 3)

            t_pred = time.perf_counter()
            proba = automl.predict_proba(X_test)
            out["predict_time_s"] = round(time.perf_counter() - t_pred, 3)

        if hasattr(proba, "values"):
            proba = proba.values
        out["proba"] = np.array(proba).tolist()
        out["best_pipeline"] = str(
            automl.get_leaderboard().iloc[0]["name"]
            if hasattr(automl, "get_leaderboard")
            else "MLJAR AutoML"
        )
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
