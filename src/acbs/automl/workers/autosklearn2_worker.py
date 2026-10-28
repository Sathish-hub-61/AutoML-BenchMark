#!/usr/bin/env python
"""Auto-sklearn 2 worker — reads TrainSpec JSON from stdin, writes RunResult JSON to stdout.

Runs inside .envs/autosklearn2/ virtualenv — Linux only.
Uses AutoSklearn2Classifier (iterative algorithms variant).
"""

from __future__ import annotations

import json
import platform
import sys
import time

import numpy as np


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

    if platform.system() == "Windows":
        out["status"] = "unsupported"
        out["error"] = "Auto-sklearn 2 requires Linux (C++ compiler + SWIG). Use WSL2 or Docker."
        out["wall_time_s"] = round(time.perf_counter() - t0, 3)
        print(json.dumps(out))
        return

    try:
        import autosklearn
        from autosklearn.experimental.askl2 import AutoSklearn2Classifier

        out["framework_version"] = autosklearn.__version__

        X_train = np.array(spec["X_train"], dtype=float)
        y_train = np.array(spec["y_train"])
        X_test = np.array(spec["X_test"], dtype=float)

        t_fit = time.perf_counter()
        clf = AutoSklearn2Classifier(
            time_left_for_this_task=spec["time_budget_s"],
            seed=spec["seed"],
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        out["fit_time_s"] = round(time.perf_counter() - t_fit, 3)

        t_pred = time.perf_counter()
        proba = clf.predict_proba(X_test)
        out["predict_time_s"] = round(time.perf_counter() - t_pred, 3)

        out["proba"] = proba.tolist()
        out["best_pipeline"] = str(clf.show_models())[:500]
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
