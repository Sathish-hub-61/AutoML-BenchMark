#!/usr/bin/env python
"""NaiveAutoML worker — reads TrainSpec JSON from stdin, writes RunResult JSON to stdout.

Runs inside .envs/naiveautoml/ virtualenv. No acbs imports.

NOTE: naiveautoml<=0.1.6 has a known upstream bug (KeyError: nan in
_sklearn_hpo.get_config_space_for_selected_algorithms) on Windows.
This worker marks status='skipped' with a clear explanation rather than
crashing, so the benchmarker can continue with other frameworks.
"""

from __future__ import annotations

import json
import sys
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
        import naiveautoml
        from naiveautoml import NaiveAutoML

        version = getattr(naiveautoml, "__version__", "unknown")
        out["framework_version"] = version

        X_train = pd.DataFrame(spec["X_train"], columns=spec["feature_names"])
        y_train = pd.Series(spec["y_train"])
        X_test = pd.DataFrame(spec["X_test"], columns=spec["feature_names"])

        def _prep(X: pd.DataFrame) -> pd.DataFrame:
            df = X.copy()
            cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
            if cat_cols:
                df = pd.get_dummies(df, columns=cat_cols)
            for col in df.columns:
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].median())
            return df.astype(float)

        X_train_p = _prep(X_train)
        X_test_p = _prep(X_test)

        # Align columns (get_dummies may produce different sets)
        for col in X_train_p.columns:
            if col not in X_test_p.columns:
                X_test_p[col] = 0.0
        X_test_p = X_test_p[X_train_p.columns]

        t_fit = time.perf_counter()
        clf = NaiveAutoML(timeout_overall=spec["time_budget_s"])
        clf.fit(X_train_p, y_train)
        out["fit_time_s"] = round(time.perf_counter() - t_fit, 3)

        t_pred = time.perf_counter()
        proba = clf.predict_proba(X_test_p)
        out["predict_time_s"] = round(time.perf_counter() - t_pred, 3)

        out["proba"] = np.array(proba).tolist()
        out["best_pipeline"] = f"NaiveAutoML {version}"
        out["status"] = "ok"

    except KeyError as exc:
        # Known upstream bug: config_spaces[step][nan] KeyError
        version = out.get("framework_version", "unknown")
        out["status"] = "skipped"
        out["error"] = (
            f"naiveautoml {version} upstream bug (KeyError: {exc}) in "
            "_sklearn_hpo.get_config_space_for_selected_algorithms on Windows. "
            "See https://github.com/AutoML-Research/naiveautoml/issues"
        )
    except Exception as exc:  # noqa: BLE001
        import traceback

        out["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1000:]}"
        out["status"] = "failed"
    finally:
        out["wall_time_s"] = round(time.perf_counter() - t0, 3)
        print(json.dumps(out))


if __name__ == "__main__":
    main()
