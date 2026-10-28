"""Regression tests for the AutoML worker adapters.

Each worker runs inside its own virtualenv, but the contract-level logic that
previously broke (probability shape, budget conversion, worker pool size) lives
in pure helpers that import without the framework installed.

Fixed defects covered here:
- h2o: ``predict`` column leaked into the probability matrix (3 cols for binary).
- lightautoml: binary predictions arrive as a single (n, 1) column.
- tpot: budgets were floored to whole minutes (20 s became 60 s).
- gama: pools of 4+ workers livelocked on spawn (0 evaluations, empty selection).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from acbs.automl.workers.gama_worker import _prep, _worker_n_jobs
from acbs.automl.workers.h2o_worker import _proba_from_preds
from acbs.automl.workers.lightautoml_worker import _aligned_proba
from acbs.automl.workers.tpot_worker import _max_time_mins, _search_limits


def test_h2o_predict_column_is_excluded():
    preds = pd.DataFrame({"predict": ["0", "1", "0"], "p0": [0.9, 0.2, 0.7], "p1": [0.1, 0.8, 0.3]})
    proba = _proba_from_preds(preds)
    assert len(proba) == 3
    assert all(len(row) == 2 for row in proba)
    assert proba[0] == [0.9, 0.1]
    assert proba[1] == [0.2, 0.8]


def test_h2o_multiclass_keeps_one_column_per_class():
    preds = pd.DataFrame(
        {
            "predict": ["a", "b"],
            "p0": [0.7, 0.1],
            "p1": [0.2, 0.6],
            "p2": [0.1, 0.3],
        }
    )
    proba = _proba_from_preds(preds)
    assert all(len(row) == 3 for row in proba)
    assert all(abs(sum(row) - 1.0) < 1e-9 for row in proba)


def test_h2o_probability_columns_are_not_alphabetically_shifted():
    # "predict" sorts before "p0"; excluding it must not reorder classes.
    preds = pd.DataFrame({"p0": [0.4], "p1": [0.6], "predict": ["1"]})
    assert _proba_from_preds(preds) == [[0.4, 0.6]]


def test_lightautoml_binary_single_column_becomes_two():
    proba = _aligned_proba([[0.7], [0.2], [0.9]], "binary")
    assert len(proba) == 3
    assert all(len(row) == 2 for row in proba)
    assert proba[0] == pytest.approx([0.3, 0.7])
    assert proba[1] == pytest.approx([0.8, 0.2])
    assert all(abs(sum(row) - 1.0) < 1e-9 for row in proba)


def test_lightautoml_binary_two_columns_pass_through():
    proba = _aligned_proba([[0.3, 0.7], [0.8, 0.2]], "binary")
    assert np.allclose(proba, [[0.3, 0.7], [0.8, 0.2]])


def test_lightautoml_multiclass_matrix_pass_through():
    matrix = np.array([[0.1, 0.6, 0.3], [0.5, 0.4, 0.1]])
    proba = _aligned_proba(matrix, "multiclass")
    assert np.allclose(proba, matrix)


def test_tpot_short_budget_is_fractional_minutes():
    assert _max_time_mins(20) == pytest.approx(20 / 60)
    assert _max_time_mins(30) == pytest.approx(0.5)
    assert _max_time_mins(60) == pytest.approx(1.0)
    assert _max_time_mins(900) == pytest.approx(15.0)


def test_tpot_sub_second_budget_floors_to_one_second():
    assert _max_time_mins(0.5) == pytest.approx(1 / 60)


def test_tpot_short_budget_shrinks_population_and_eval_cap():
    """Generation 0 has no time check: an untuned default overshoots the wall limit.

    50 individuals x 10-fold with a 10-minute eval cap got a 60 s budget killed
    at 120 s on german_credit; the limits must scale down with the budget.
    """
    t_mins, eval_mins, pop = _search_limits(60)
    assert t_mins == pytest.approx(1.0)
    assert pop == 10
    # Worst case gen-0 cost stays under budget + grace (120 s), incl. TPOT's 1.25x cap factor.
    assert pop * eval_mins * 60 * 1.25 <= 120

    assert _search_limits(300)[2] == 25
    assert _search_limits(900)[2] == 50
    assert _search_limits(900)[0] == pytest.approx(15.0)
    # Eval cap grows with the budget but never exceeds a sixth of it (floor 4 s).
    assert _search_limits(480)[1] == pytest.approx(60 / 60, abs=1e-6)


def test_gama_worker_pool_capped_at_two(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    assert _worker_n_jobs() == 2
    monkeypatch.setenv("OMP_NUM_THREADS", "6")
    assert _worker_n_jobs() == 2  # 6 is the default pool that livelocks
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    assert _worker_n_jobs() == 2


def test_gama_honours_smaller_thread_limits(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    assert _worker_n_jobs() == 1
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    assert _worker_n_jobs() == 2


def test_gama_invalid_thread_env_falls_back_to_cap(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "abc")
    assert _worker_n_jobs() == 2


def test_gama_prep_encodes_categoricals_and_fills_missing():
    X = pd.DataFrame(
        {
            "num": [1.0, np.nan, 3.0, 4.0],
            "cat": ["a", "b", "a", "b"],
        }
    )
    out = _prep(X)
    assert out.shape == (4, 3)  # num + two one-hot columns
    assert not out.isna().any().any()
    assert all(pd.api.types.is_float_dtype(t) for t in out.dtypes)


def test_gama_prep_outputs_string_column_names():
    """Mixed int/str names make sklearn reject every GAMA evaluation.

    GAMA re-infers 0/1 columns as categorical and its encoding pipeline keeps
    int base names, so the worker must hand over uniformly string names.
    """
    X = pd.DataFrame({0: [1.0, 2.0], "flag": [0, 1], "cat": ["a", "b"]})
    out = _prep(X)
    assert all(isinstance(c, str) for c in out.columns)
    assert "flag_0" in " ".join(out.columns) or any("flag" in c for c in out.columns)
