"""Live availability of AutoML framework environments.

Answers, for every canonical framework id: is its isolated venv installed, is it
supported on this OS, and did its last compatibility check fail? Both the API
route and the automatic selector consume this — it is business logic, so it
lives here rather than in the route.
"""

from __future__ import annotations

import json
import platform

import yaml

from acbs.utils.paths import repo_root

# All canonical framework IDs.
CANONICAL_FRAMEWORK_IDS = [
    "autogluon",
    "flaml",
    "gama",
    "h2o",
    "lightautoml",
    "mljar",
    "naiveautoml",
    "tpot",
]


def _framework_meta() -> dict:
    info_file = repo_root() / "configs" / "frameworks_info.yaml"
    if not info_file.exists():
        return {}
    try:
        with open(info_file, encoding="utf-8") as fh:
            return yaml.safe_load(fh).get("frameworks", {}) or {}
    except Exception:
        return {}


def _compat_cache() -> dict:
    compat_file = repo_root() / "results" / "compat.json"
    if not compat_file.exists():
        return {}
    try:
        with open(compat_file, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def framework_availability() -> list[dict]:
    """Return availability records for every canonical framework id.

    Each record: ``id``, ``name``, ``strategy``, ``kind``, ``installed``,
    ``version``, ``os_supported``, ``reason``.
    """
    root = repo_root()
    is_windows = platform.system() == "Windows"
    meta_all = _framework_meta()
    compat_all = _compat_cache()

    results = []
    for fid in CANONICAL_FRAMEWORK_IDS:
        meta = meta_all.get(fid, {})
        compat = compat_all.get(fid, {})

        env_dir = root / ".envs" / fid
        exe = (
            env_dir
            / ("Scripts" if is_windows else "bin")
            / ("python.exe" if is_windows else "python")
        )
        installed = exe.exists()

        os_supported = True
        reason = None

        if not installed:
            reason = f"Virtualenv not configured under .envs/{fid}/. Run scripts/setup_envs.py"
        elif compat.get("status") == "failed":
            reason = compat.get("error") or "Framework compatibility test failed."

        results.append(
            {
                "id": fid,
                "name": meta.get("name", fid.title()),
                "strategy": meta.get("strategy", "Automated machine learning optimization"),
                "kind": meta.get("kind", "single_pipeline"),
                "installed": installed,
                "version": compat.get("version"),
                "os_supported": os_supported,
                "reason": reason,
            }
        )
    return results
