"""Tests for the AutoML runner, contracts, and error isolation."""

from __future__ import annotations

import sys
import textwrap
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from acbs.automl.contracts import RunResult, TrainSpec
from acbs.automl.evaluate import evaluate_automl_result
from acbs.automl.runner import run_framework


def test_train_spec_serialization():
    spec = TrainSpec(
        framework_id="flaml",
        X_train=[[1.0, 2.0], [3.0, 4.0]],
        y_train=[0, 1],
        X_test=[[5.0, 6.0]],
        feature_names=["f1", "f2"],
        task_type="binary",
        n_classes=2,
        time_budget_s=30,
        seed=42,
    )
    raw = spec.to_json()
    assert '"framework_id": "flaml"' in raw
    assert '"time_budget_s": 30' in raw


def test_run_result_deserialization():
    raw_json = (
        '{"status": "ok", "proba": [[0.1, 0.9]], "fit_time_s": 1.2, "framework_version": "1.0"}'
    )
    res = RunResult.from_json("flaml", raw_json)
    assert res.status == "ok"
    assert res.proba == [[0.1, 0.9]]
    assert res.fit_time_s == 1.2
    assert res.framework_version == "1.0"

    bad_json = "not a valid json"
    res_bad = RunResult.from_json("flaml", bad_json)
    assert res_bad.status == "failed"
    assert "JSON decode error" in (res_bad.error or "")


def test_unsupported_framework_on_windows():
    spec = TrainSpec(
        framework_id="autosklearn",
        X_train=[[1.0, 2.0]],
        y_train=[0],
        X_test=[[3.0, 4.0]],
        feature_names=["f1", "f2"],
        task_type="binary",
        n_classes=2,
        time_budget_s=10,
    )
    if sys.platform == "win32":
        res = run_framework(spec)
        assert res.status == "unsupported"
        assert "Linux" in (res.error or "")


def test_missing_venv():
    spec = TrainSpec(
        framework_id="nonexistent_framework_xyz",
        X_train=[[1.0, 2.0]],
        y_train=[0],
        X_test=[[3.0, 4.0]],
        feature_names=["f1", "f2"],
        task_type="binary",
        n_classes=2,
        time_budget_s=10,
    )
    res = run_framework(spec)
    assert res.status == "failed"
    assert "venv not found" in (res.error or "")


@pytest.fixture
def fake_worker_dir(tmp_path: Path):
    """Create fake worker scripts simulating different behaviors."""
    ok_script = tmp_path / "fake_ok_worker.py"
    ok_script.write_text(
        textwrap.dedent("""
            import json, sys
            spec = json.loads(sys.stdin.read())
            n_test = len(spec["X_test"])
            out = {
                "status": "ok",
                "proba": [[0.2, 0.8] for _ in range(n_test)],
                "fit_time_s": 0.5,
                "predict_time_s": 0.05,
                "framework_version": "fake-1.0",
                "best_pipeline": "FakeEstimator()",
                "error": None,
            }
            print(json.dumps(out))
        """),
        encoding="utf-8",
    )

    crash_script = tmp_path / "fake_crash_worker.py"
    crash_script.write_text(
        textwrap.dedent("""
            import sys
            sys.stderr.write("Fatal exception: GPU out of memory\\n")
            sys.exit(1)
        """),
        encoding="utf-8",
    )

    timeout_script = tmp_path / "fake_timeout_worker.py"
    timeout_script.write_text(
        textwrap.dedent("""
            import time
            time.sleep(10)
        """),
        encoding="utf-8",
    )

    return {
        "ok": ok_script,
        "crash": crash_script,
        "timeout": timeout_script,
    }


def test_runner_with_fake_ok(fake_worker_dir):
    spec = TrainSpec(
        framework_id="fake_ok",
        X_train=[[1.0, 2.0], [3.0, 4.0]],
        y_train=[0, 1],
        X_test=[[5.0, 6.0], [7.0, 8.0]],
        feature_names=["f1", "f2"],
        task_type="binary",
        n_classes=2,
        time_budget_s=5,
    )
    with (
        patch("acbs.automl.runner._venv_python", return_value=Path(sys.executable)),
        patch("acbs.automl.runner._worker_script_path", return_value=fake_worker_dir["ok"]),
    ):
        res = run_framework(spec)
        assert res.status == "ok"
        assert res.proba == [[0.2, 0.8], [0.2, 0.8]]
        assert res.framework_version == "fake-1.0"
        assert res.best_pipeline == "FakeEstimator()"


def test_runner_crash_isolation(fake_worker_dir):
    spec = TrainSpec(
        framework_id="fake_crash",
        X_train=[[1.0, 2.0]],
        y_train=[0],
        X_test=[[3.0, 4.0]],
        feature_names=["f1", "f2"],
        task_type="binary",
        n_classes=2,
        time_budget_s=5,
    )
    with (
        patch("acbs.automl.runner._venv_python", return_value=Path(sys.executable)),
        patch("acbs.automl.runner._worker_script_path", return_value=fake_worker_dir["crash"]),
    ):
        res = run_framework(spec)
        assert res.status == "failed"
        assert "GPU out of memory" in (res.error or "")


def test_runner_timeout_killed(fake_worker_dir):
    spec = TrainSpec(
        framework_id="fake_timeout",
        X_train=[[1.0, 2.0]],
        y_train=[0],
        X_test=[[3.0, 4.0]],
        feature_names=["f1", "f2"],
        task_type="binary",
        n_classes=2,
        time_budget_s=1,
    )
    # Set grace_extra_s = 0 so total timeout is 1s, less than the 10s sleep
    with (
        patch("acbs.automl.runner._venv_python", return_value=Path(sys.executable)),
        patch("acbs.automl.runner._worker_script_path", return_value=fake_worker_dir["timeout"]),
    ):
        res = run_framework(spec, grace_extra_s=0)
        assert res.status == "timeout"
        assert "timed out" in (res.error or "")


def test_timeout_kills_grandchild_pipe_holders(tmp_path: Path):
    """A worker whose child holds the output pipes must not stall past its wall limit.

    subprocess.run(timeout=) waits for pipe EOF, which grandchildren keep open —
    this measured 397 s for a 70 s limit before the runner killed the tree.
    """
    orphan_script = tmp_path / "fake_orphan_worker.py"
    orphan_script.write_text(
        textwrap.dedent(
            """
            import subprocess, sys, time
            subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            time.sleep(60)
            """
        ),
        encoding="utf-8",
    )
    spec = TrainSpec(
        framework_id="fake_orphan",
        X_train=[[1.0, 2.0]],
        y_train=[0],
        X_test=[[3.0, 4.0]],
        feature_names=["f1", "f2"],
        task_type="binary",
        n_classes=2,
        time_budget_s=1,
    )
    with (
        patch("acbs.automl.runner._venv_python", return_value=Path(sys.executable)),
        patch("acbs.automl.runner._worker_script_path", return_value=orphan_script),
    ):
        t0 = time.perf_counter()
        res = run_framework(spec, grace_extra_s=0)
        elapsed = time.perf_counter() - t0

    assert res.status == "timeout"
    # 1 s limit + tree kill + a bounded 15 s pipe drain — never the child's 60 s
    assert elapsed < 20, f"runner stalled for {elapsed:.1f}s"


def test_evaluate_automl_result():
    res = RunResult(
        framework_id="fake",
        status="ok",
        proba=[[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]],
    )
    eval_res = evaluate_automl_result(res, y_test=[1, 0, 1], classes=[0, 1])
    assert eval_res["y_pred"] == [1, 0, 1]
    assert eval_res["test_metrics"]["accuracy"] == 1.0
    assert eval_res["test_metrics"]["f1_macro"] == 1.0


def test_evaluate_automl_shape_mismatch():
    res = RunResult(
        framework_id="fake",
        status="ok",
        proba=[[0.1, 0.9]],  # 1 row, but y_test has 2 rows
    )
    with pytest.raises(ValueError, match="Prediction sample count mismatch"):
        evaluate_automl_result(res, y_test=[1, 0], classes=[0, 1])

    res_col_mismatch = RunResult(
        framework_id="fake",
        status="ok",
        proba=[[0.1, 0.9]],  # 2 columns, but classes has 3
    )
    with pytest.raises(ValueError, match="Probability column count mismatch"):
        evaluate_automl_result(res_col_mismatch, y_test=[1], classes=[0, 1, 2])
