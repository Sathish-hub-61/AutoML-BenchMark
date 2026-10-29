"""Host-side runner — spawns each framework worker in its isolated venv.

Responsibilities
----------------
- Resolve the venv python for a given framework ID.
- Inject the per-framework worker script as -c code.
- Serialise TrainSpec → stdin, deserialise stdout → RunResult.
- Enforce wall-clock budget (time_budget_s + 60 s grace).
- Never raise — failed/timeout/unsupported are all captured as RunResult.
"""

from __future__ import annotations

import logging
import os
import platform
import signal
import subprocess
import time
from pathlib import Path

from acbs.automl.contracts import RunResult, TrainSpec

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    """Return the repository root (two levels above src/)."""
    return Path(__file__).resolve().parent.parent.parent.parent


def _venv_python(framework_id: str) -> Path | None:
    """Return the python executable for a framework's isolated venv, or None."""
    root = _repo_root()
    env_dir = root / ".envs" / framework_id
    exe = (
        env_dir
        / ("Scripts" if platform.system() == "Windows" else "bin")
        / ("python.exe" if platform.system() == "Windows" else "python")
    )
    return exe if exe.exists() else None


def _worker_script_path(framework_id: str) -> Path | None:
    """Return the path to the per-framework worker script, or None if not found."""
    workers_dir = Path(__file__).parent / "workers"
    candidate = workers_dir / f"{framework_id}_worker.py"
    return candidate if candidate.exists() else None


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill a worker and every child it spawned.

    Workers fork grandchildren (framework process pools) that inherit the
    stdout/stderr pipes; killing only the direct child never releases those
    pipes, so any wait on them blocks until the grandchildren exit on their
    own — a real LightAutoML run held a 70 s limit open for 397 s that way.
    """
    try:
        if platform.system() == "Windows":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                timeout=10,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def _run_worker(cmd: list[str], spec_json: str, wall_limit: int) -> tuple[str, str, bool]:
    """Run a worker under a hard wall-clock limit.

    Returns ``(stdout, stderr, timed_out)``. Unlike ``subprocess.run(timeout=)``
    — whose Windows timeout path waits on pipe EOF held by grandchildren —
    this kills the whole process tree and bounds the final pipe read, so a
    worker with lingering children can never stall the benchmark.
    """
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name == "posix",
    )
    try:
        out, err = proc.communicate(input=spec_json, timeout=wall_limit)
        return out or "", err or "", False
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            # Retrying after TimeoutExpired keeps whatever was already read —
            # a worker that printed its result before hanging still counts.
            out, err = proc.communicate(timeout=15)
        except Exception:  # noqa: BLE001
            out, err = "", ""
        return out or "", err or "", True


def run_framework(
    spec: TrainSpec,
    *,
    grace_extra_s: int = 60,
    backend: str = "venv",
) -> RunResult:
    """Run a single AutoML framework in its isolated venv or docker container.

    Parameters
    ----------
    spec:
        Fully populated TrainSpec for this framework.
    grace_extra_s:
        Extra seconds beyond spec.time_budget_s before the subprocess is killed.
    backend:
        'venv' (default) or 'docker'.

    Returns
    -------
    RunResult
        Never raises; errors are captured inside RunResult.
    """
    fid = spec.framework_id
    wall_limit = spec.time_budget_s + grace_extra_s
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Docker backend execution path
    # ------------------------------------------------------------------
    if backend == "docker":
        cmd = [
            "docker",
            "run",
            "--rm",
            "-i",
            f"acbs-fw-{fid}:latest",
            "python",
            "-m",
            f"acbs.automl.workers.{fid}_worker",
        ]
        spec_json = spec.to_json()
        logger.info("[%s] launching docker worker: %s", fid, " ".join(cmd))
        try:
            out, err, timed_out = _run_worker(cmd, spec_json, wall_limit)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[%s] error launching docker worker", fid)
            return RunResult.failed(fid, f"Docker backend error: {exc}")

        elapsed_s = time.perf_counter() - t0
        if timed_out:
            logger.warning("[%s] docker worker timed out after %ds", fid, wall_limit)
            return RunResult.timeout(fid, spec.time_budget_s)

        out = (out or "").strip()
        if not out:
            err_msg = (err or "").strip() or "No output on stdout from docker worker"
            return RunResult.failed(fid, err_msg[-500:], elapsed_s)

        try:
            return RunResult.from_json(fid, out)
        except Exception as exc:  # noqa: BLE001
            return RunResult.failed(fid, f"JSON parse error from docker worker: {exc}", elapsed_s)

    # ------------------------------------------------------------------
    # Virtualenv backend availability gate
    # ------------------------------------------------------------------
    python_exe = _venv_python(fid)
    if python_exe is None:
        return RunResult.failed(
            fid, f"venv not found under .envs/{fid}/. Run scripts/setup_envs.py first."
        )

    worker_script = _worker_script_path(fid)
    if worker_script is None:
        return RunResult.failed(
            fid, f"Worker script not found: src/acbs/automl/workers/{fid}_worker.py"
        )

    # ------------------------------------------------------------------
    # Subprocess execution
    # ------------------------------------------------------------------
    spec_json = spec.to_json()
    cmd = [str(python_exe), str(worker_script)]

    logger.info(
        "[%s] launching worker (budget=%ds, wall_limit=%ds)", fid, spec.time_budget_s, wall_limit
    )
    try:
        out, err, timed_out = _run_worker(cmd, spec_json, wall_limit)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[%s] unexpected error launching worker", fid)
        return RunResult.failed(fid, f"{type(exc).__name__}: {exc}", time.perf_counter() - t0)

    wall_time_s = round(time.perf_counter() - t0, 2)

    # ------------------------------------------------------------------
    # Parse output
    # ------------------------------------------------------------------
    # Worker must print exactly one JSON line to stdout. A worker that already
    # printed its result before being killed keeps it (partial output survives
    # the timeout path).
    json_lines = [line.strip() for line in out.splitlines() if line.strip().startswith("{")]
    if timed_out and not json_lines:
        logger.warning("[%s] worker killed after wall limit %ds", fid, wall_limit)
        return RunResult.timeout(fid, spec.time_budget_s)
    if not json_lines:
        stderr_preview = (err or "").strip()[-500:]
        return RunResult.failed(
            fid,
            f"No JSON output from worker. stderr: {stderr_preview}",
            wall_time_s,
        )

    result = RunResult.from_json(fid, json_lines[-1])
    result.wall_time_s = wall_time_s
    logger.info(
        "[%s] finished status=%s fit=%.1fs predict=%.1fs wall=%.1fs",
        fid,
        result.status,
        result.fit_time_s,
        result.predict_time_s,
        result.wall_time_s,
    )
    return result
