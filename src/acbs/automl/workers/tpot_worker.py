#!/usr/bin/env python
"""TPOT worker — reads TrainSpec JSON from stdin, writes RunResult JSON to stdout.

Runs inside .envs/tpot/ virtualenv. No acbs imports.
TPOT uses genetic programming for pipeline optimisation.
"""

from __future__ import annotations

import json
import sys
import time

import pandas as pd


def _prep(X: pd.DataFrame, y: pd.Series | None = None):
    """TPOT requires all-numeric input. Encode categoricals, fill NaN."""
    df = X.copy()
    cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols)
    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df.astype(float)


def _max_time_mins(time_budget_s: float) -> float:
    """Convert the wall-clock budget to TPOT's ``max_time_mins``.

    TPOT accepts fractional minutes (``time.time() + max_time_mins * 60``),
    so short budgets pass through exactly; flooring them to 1 minute would
    inflate a 20 s budget to 60 s and overshoot the runner's own timeout.
    """
    return max(1, time_budget_s) / 60.0


def _search_limits(time_budget_s: float) -> tuple[float, float, int]:
    """Scale TPOT's search knobs to the wall-clock budget.

    Returns ``(max_time_mins, max_eval_time_mins, population_size)``.

    TPOT only checks its timer *between* generations: generation 0 evaluates the
    whole initial population (population_size x internal cv folds) with no time
    check at all, so a short budget must also bound the per-evaluation cap and
    the initial population — otherwise the runner's wall limit (budget + grace)
    kills the worker (observed: 60 s budget killed at 120 s on german_credit
    with TPOT's defaults of 50 x 10-fold).
    """
    budget_s = max(1, time_budget_s)
    max_time_mins = _max_time_mins(budget_s)
    max_eval_time_mins = round(max(4.0, budget_s / 8) / 60.0, 3)
    population_size = 50 if budget_s >= 600 else (25 if budget_s >= 240 else 10)
    return max_time_mins, max_eval_time_mins, population_size


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
        import tpot
        from tpot import TPOTClassifier

        out["framework_version"] = tpot.__version__

        X_train = pd.DataFrame(spec["X_train"], columns=spec["feature_names"])
        y_train = pd.Series(spec["y_train"])
        X_test = pd.DataFrame(spec["X_test"], columns=spec["feature_names"])

        X_train_p = _prep(X_train)
        X_test_p = _prep(X_test)

        # Ensure test columns match train (get_dummies may differ)
        for col in X_train_p.columns:
            if col not in X_test_p.columns:
                X_test_p[col] = 0.0
        X_test_p = X_test_p[X_train_p.columns]

        max_time_mins, max_eval_time_mins, population_size = _search_limits(spec["time_budget_s"])

        t_fit = time.perf_counter()
        clf = TPOTClassifier(
            max_time_mins=max_time_mins,
            max_eval_time_mins=max_eval_time_mins,
            population_size=population_size,
            cv=5,
            random_state=spec["seed"],
            verbose=0,
            n_jobs=1,
        )
        clf.fit(X_train_p.values, y_train.values)
        out["fit_time_s"] = round(time.perf_counter() - t_fit, 3)

        t_pred = time.perf_counter()
        proba = clf.predict_proba(X_test_p.values)
        out["predict_time_s"] = round(time.perf_counter() - t_pred, 3)

        out["proba"] = proba.tolist()
        out["best_pipeline"] = str(clf.fitted_pipeline_)
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
