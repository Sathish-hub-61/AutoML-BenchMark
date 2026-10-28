#!/usr/bin/env python
"""H2O AutoML worker — reads TrainSpec JSON from stdin, writes RunResult JSON to stdout.

Runs inside .envs/h2o/ virtualenv. No acbs imports.
H2O starts a local JVM cluster; we shut it down in finally{}.
"""

from __future__ import annotations

import json
import sys
import time

import pandas as pd


def _proba_from_preds(preds_df: pd.DataFrame) -> list:
    """Extract the per-class probability columns from an H2O prediction frame.

    H2O emits ``predict`` plus ``p0, p1, ...`` (alphabetical class order).
    ``predict`` also starts with "p", so a naive prefix match includes it and
    makes binary runs emit 3 columns, which fails central scoring.
    """
    proba_cols = [c for c in preds_df.columns if c.startswith("p") and c != "predict"]
    return preds_df[proba_cols].values.tolist()


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
    h2o = None

    try:
        import h2o
        from h2o.automl import H2OAutoML

        out["framework_version"] = h2o.__version__

        X_train = pd.DataFrame(spec["X_train"], columns=spec["feature_names"])
        y_train = pd.Series(spec["y_train"], name="target").astype(str)
        X_test = pd.DataFrame(spec["X_test"], columns=spec["feature_names"])

        df_train = X_train.copy()
        df_train["target"] = y_train

        h2o.init(nthreads=-1, max_mem_size="2G", port=54321, verbose=False)
        hf_train = h2o.H2OFrame(df_train)
        hf_train["target"] = hf_train["target"].asfactor()
        hf_test = h2o.H2OFrame(X_test)

        t_fit = time.perf_counter()
        aml = H2OAutoML(
            max_runtime_secs=spec["time_budget_s"],
            seed=spec["seed"],
            verbosity=None,
        )
        aml.train(y="target", training_frame=hf_train)
        out["fit_time_s"] = round(time.perf_counter() - t_fit, 3)

        t_pred = time.perf_counter()
        preds_h2o = aml.predict(hf_test)
        out["predict_time_s"] = round(time.perf_counter() - t_pred, 3)

        out["proba"] = _proba_from_preds(preds_h2o.as_data_frame())

        lb = aml.leaderboard.as_data_frame()
        out["best_pipeline"] = lb.iloc[0]["model_id"] if len(lb) > 0 else "unknown"
        out["status"] = "ok"

    except Exception as exc:  # noqa: BLE001
        import traceback

        out["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1000:]}"
        out["status"] = "failed"
    finally:
        if h2o is not None:
            try:
                h2o.cluster().shutdown(prompt=False)
            except Exception:
                pass
        out["wall_time_s"] = round(time.perf_counter() - t0, 3)
        print(json.dumps(out))


if __name__ == "__main__":
    main()
