from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from acbs.benchmark.engine import BenchmarkResult


def results_to_dataframe(result: BenchmarkResult) -> pd.DataFrame:
    rows = []
    for m in result.models:
        row = {
            "model_name": m.model_name,
            "status": m.status,
            "notes": m.notes,
            "error": m.error,
            "training_time": m.training_time,
            "prediction_time_per_1k": m.prediction_time_per_1k,
            "interpretability_score": m.interpretability_score,
            "best_params": m.best_params,
        }
        for k, v in m.test_metrics.items():
            row[k] = v
        for k, v in m.cv_metrics.items():
            row[f"cv_{k}"] = v
            row[f"cv_{k}_std"] = m.cv_std.get(k)
        rows.append(row)
    return pd.DataFrame(rows)


def fold_score_matrix(result: BenchmarkResult, metric: str = "f1_macro") -> pd.DataFrame:
    data = {}
    for m in result.models:
        if m.status == "ok" and metric in m.fold_scores:
            data[m.model_name] = m.fold_scores[metric]
    return pd.DataFrame(data)


def rank_by_metric(df: pd.DataFrame, metric: str, *, higher_is_better: bool = True) -> pd.Series:
    ok = df["status"].eq("ok") if "status" in df.columns else pd.Series(True, index=df.index)
    series = df.loc[ok, metric]
    return series.rank(ascending=not higher_is_better, method="average")


def _nadeau_bengio_ttest(a: np.ndarray, b: np.ndarray, n_train: int, n_test: int) -> float:
    diff = a - b
    n = len(diff)
    mean = diff.mean()
    var = diff.var(ddof=1)
    corr = n_test / n_train
    denom = np.sqrt((1 / n + corr) * var)
    if denom == 0:
        return 1.0
    t = mean / denom
    p = 2 * stats.t.sf(abs(t), df=n - 1)
    return float(p)


def compare_models_statistically(
    fold_scores: pd.DataFrame,
    *,
    n_train: int = 80,
    n_test: int = 20,
    alpha: float = 0.05,
) -> dict:
    cols = [c for c in fold_scores.columns]
    if len(cols) < 2:
        return {"p_values": {}, "statistically_tied_with_best": []}
    means = fold_scores.mean()
    best = means.idxmax()
    friedman_p = None
    if len(cols) >= 3 and len(fold_scores) >= 5:
        try:
            friedman_p = float(stats.friedmanchisquare(*[fold_scores[c] for c in cols]).pvalue)
        except Exception:
            friedman_p = None
    p_values = {}
    tied = [best]
    best_scores = fold_scores[best].to_numpy()
    for col in cols:
        if col == best:
            continue
        other = fold_scores[col].to_numpy()
        try:
            p = _nadeau_bengio_ttest(best_scores, other, n_train, n_test)
        except Exception:
            try:
                p = float(stats.wilcoxon(best_scores, other, zero_method="zsplit").pvalue)
            except Exception:
                p = 1.0
        p_values[f"{best}_vs_{col}"] = p
        if p >= alpha:
            tied.append(col)
    return {
        "friedman_p": friedman_p,
        "p_values": p_values,
        "best": best,
        "statistically_tied_with_best": tied,
    }


def average_rank_across_datasets(tables: list[pd.DataFrame], metric: str = "f1_macro") -> pd.Series:
    ranks = []
    for df in tables:
        ok = df[df.get("status", "ok") == "ok"] if "status" in df.columns else df
        ranks.append(ok.set_index("model_name")[metric].rank(ascending=False, method="average"))
    return pd.concat(ranks, axis=1).mean(axis=1).sort_values()
