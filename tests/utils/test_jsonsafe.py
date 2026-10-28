"""Tests for acbs.utils.jsonsafe."""

from __future__ import annotations

import json
import math

import numpy as np

from acbs.utils.jsonsafe import json_safe


def test_nan_and_inf_become_null():
    out = json_safe({"a": float("nan"), "b": float("inf"), "c": float("-inf"), "d": 1.5})
    assert out == {"a": None, "b": None, "c": None, "d": 1.5}
    # the exact condition that used to 500: Starlette's allow_nan=False
    json.dumps(out, allow_nan=False)


def test_nested_containers_are_sanitised():
    out = json_safe([{"m": [float("nan"), np.float64(0.5)]}, (float("nan"),)])
    assert out == [{"m": [None, 0.5]}, [None]]
    json.dumps(out, allow_nan=False)


def test_numpy_scalars_become_python_scalars():
    out = json_safe(
        {
            "i": np.int64(7),
            "f": np.float32(0.25),
            "b": np.bool_(True),
        }
    )
    assert out == {"i": 7, "f": 0.25, "b": True}
    json.dumps(out, allow_nan=False)


def test_plain_values_pass_through():
    assert json_safe(None) is None
    assert json_safe("x") == "x"
    assert json_safe(3) == 3
    assert json_safe(True) is True


def test_unknown_objects_stringify_instead_of_raising():
    class Weird:
        def __repr__(self) -> str:
            raise RuntimeError("nope")

    out = json_safe(Weird())
    assert isinstance(out, str)


def test_failed_model_row_shape():
    # what a failed model contributes to the score table
    row = {
        "model_name": "flaml",
        "status": "failed",
        "f1_macro": math.nan,
        "roc_auc": math.nan,
        "training_time": None,
    }
    json.dumps(json_safe([row]), allow_nan=False)
