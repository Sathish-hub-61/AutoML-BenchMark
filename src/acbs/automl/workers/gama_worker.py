#!/usr/bin/env python
"""GAMA worker — reads TrainSpec JSON from stdin, writes RunResult JSON to stdout.

Runs inside .envs/gama/ virtualenv. No acbs imports.
GAMA uses evolutionary search over scikit-learn pipelines.
"""

from __future__ import annotations

import json
import os
import sys
import time

import pandas as pd


def _prep(X: pd.DataFrame) -> pd.DataFrame:
    """Prepare all-numeric input for GAMA with string column names.

    Categoricals are one-hot encoded and NaNs filled with the column median.
    The result stays a DataFrame with every column name cast to ``str``:
    GAMA wraps plain arrays into int-named frames, re-infers 0/1 columns as
    categorical, and its own encoding pipeline then emits a mix of int and str
    column names — which sklearn rejects for every evaluation
    (``Feature names ... has ['int', 'str']``), collapsing the population to
    fewer than 3 individuals. GAMA's own loaders always use string names.
    """
    df = X.copy()
    cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols)
    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    df = df.astype(float)
    df.columns = [str(c) for c in df.columns]
    return df


def _worker_n_jobs() -> int:
    """Return the GAMA evaluation pool size, capped at 2 workers.

    GAMA's pool busy-polls; pools of 4+ on this platform let one child die
    during spawn (EOFError) so no evaluation completes and fit crashes with
    an empty-selection IndexError (measured: 3-4/3-4 failing at 4 and 6
    workers, 0/0 failing at 1 and 2). Honour a smaller OMP_NUM_THREADS.
    """
    env_jobs = os.environ.get("OMP_NUM_THREADS", "")
    if env_jobs.isdigit() and int(env_jobs) > 0:
        return min(2, int(env_jobs))
    return 2


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
        import gama as _gama_mod
        from gama import GamaClassifier

        out["framework_version"] = getattr(_gama_mod, "__version__", "unknown")

        X_train = pd.DataFrame(spec["X_train"], columns=spec["feature_names"])
        y_train = pd.Series(spec["y_train"])
        X_test = pd.DataFrame(spec["X_test"], columns=spec["feature_names"])

        X_train_arr = _prep(X_train)
        X_test_arr = _prep(X_test)
        # One-hot columns may differ between frames; align test to train.
        X_test_arr = X_test_arr.reindex(columns=X_train_arr.columns, fill_value=0.0)

        t_fit = time.perf_counter()
        n_jobs = _worker_n_jobs()
        clf = GamaClassifier(
            max_total_time=spec["time_budget_s"],
            random_state=spec["seed"],
            n_jobs=n_jobs,
            verbosity=0,
        )
        clf.fit(X_train_arr, y_train.values)
        out["fit_time_s"] = round(time.perf_counter() - t_fit, 3)

        t_pred = time.perf_counter()
        proba = clf.predict_proba(X_test_arr)
        out["predict_time_s"] = round(time.perf_counter() - t_pred, 3)

        out["proba"] = proba.tolist()
        out["best_pipeline"] = str(clf.model)
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
