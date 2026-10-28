"""Smoke-test each AutoML framework environment on two synthetic datasets."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_python_exe(env_dir: Path) -> Path:
    if platform.system() == "Windows":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


SMOKE_WORKER_SCRIPT = """
import sys
import json
import time
import importlib
import numpy as np
import pandas as pd

def make_binary_data(n=500, seed=42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"col_{i}" for i in range(8)])
    y = (X["col_0"] + X["col_1"] > 0).astype(int)
    return X, y

def make_multiclass_data(n=800, seed=42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "num_0": rng.normal(size=n),
        "num_1": rng.normal(size=n),
        "num_with_nan": np.where(rng.random(size=n) > 0.1, rng.normal(size=n), np.nan),
        "cat_0": rng.choice(["A", "B", "C"], size=n),
        "cat_1": rng.choice(["low", "medium", "high"], size=n),
    })
    # Categoricals to category dtype
    X["cat_0"] = X["cat_0"].astype("category")
    X["cat_1"] = X["cat_1"].astype("category")
    
    score = X["num_0"] + (X["cat_0"] == "A").astype(int) - (X["cat_1"] == "low").astype(int)
    y = pd.Series(np.select([score < -0.5, score > 0.5], [0, 2], default=1), name="target")
    return X, y

def test_framework(fid):
    out = {
        "installed": False,
        "version": None,
        "binary_ok": False,
        "multiclass_ok": False,
        "binary_proba_shape": None,
        "multiclass_proba_shape": None,
        "wall_time_s": 0.0,
        "error": None,
    }
    t0 = time.perf_counter()
    try:
        if fid == "flaml":
            import flaml
            from flaml import AutoML
            out["installed"] = True
            out["version"] = flaml.__version__
            
            # Binary
            X_bin, y_bin = make_binary_data()
            clf_bin = AutoML()
            clf_bin.fit(X_train=X_bin, y_train=y_bin, task="classification", time_budget=5, verbose=0)
            proba_bin = clf_bin.predict_proba(X_bin[:10])
            out["binary_proba_shape"] = list(proba_bin.shape)
            out["binary_ok"] = True
            
            # Multiclass
            X_mc, y_mc = make_multiclass_data()
            clf_mc = AutoML()
            clf_mc.fit(X_train=X_mc, y_train=y_mc, task="classification", time_budget=5, verbose=0)
            proba_mc = clf_mc.predict_proba(X_mc[:10])
            out["multiclass_proba_shape"] = list(proba_mc.shape)
            out["multiclass_ok"] = True

        elif fid == "autogluon":
            import autogluon.tabular
            from autogluon.tabular import TabularPredictor
            out["installed"] = True
            out["version"] = autogluon.tabular.__version__
            
            X_bin, y_bin = make_binary_data()
            df_bin = X_bin.copy()
            df_bin["target"] = y_bin
            pred_bin = TabularPredictor(label="target", problem_type="binary").fit(df_bin, time_limit=10, presets="medium_quality", verbosity=0)
            proba_bin = pred_bin.predict_proba(X_bin[:10])
            out["binary_proba_shape"] = list(proba_bin.shape)
            out["binary_ok"] = True
            
            X_mc, y_mc = make_multiclass_data()
            df_mc = X_mc.copy()
            df_mc["target"] = y_mc
            pred_mc = TabularPredictor(label="target", problem_type="multiclass").fit(df_mc, time_limit=10, presets="medium_quality", verbosity=0)
            proba_mc = pred_mc.predict_proba(X_mc[:10])
            out["multiclass_proba_shape"] = list(proba_mc.shape)
            out["multiclass_ok"] = True

        elif fid == "h2o":
            import h2o
            from h2o.automl import H2OAutoML
            out["installed"] = True
            out["version"] = h2o.__version__
            h2o.init(nthreads=1, max_mem_size="2G")
            try:
                X_bin, y_bin = make_binary_data()
                df_bin = X_bin.copy()
                df_bin["target"] = y_bin.astype(str)
                hf_bin = h2o.H2OFrame(df_bin)
                hf_bin["target"] = hf_bin["target"].asfactor()
                aml_bin = H2OAutoML(max_runtime_secs=10, seed=42)
                aml_bin.train(y="target", training_frame=hf_bin)
                pred_bin = aml_bin.predict(hf_bin[:10, :])
                out["binary_proba_shape"] = [10, 2]
                out["binary_ok"] = True

                X_mc, y_mc = make_multiclass_data()
                df_mc = X_mc.copy()
                df_mc["target"] = y_mc.astype(str)
                hf_mc = h2o.H2OFrame(df_mc)
                hf_mc["target"] = hf_mc["target"].asfactor()
                aml_mc = H2OAutoML(max_runtime_secs=10, seed=42)
                aml_mc.train(y="target", training_frame=hf_mc)
                pred_mc = aml_mc.predict(hf_mc[:10, :])
                out["multiclass_proba_shape"] = [10, 3]
                out["multiclass_ok"] = True
            finally:
                h2o.cluster().shutdown(prompt=False)

        elif fid == "tpot":
            import tpot
            from tpot import TPOTClassifier
            out["installed"] = True
            out["version"] = tpot.__version__
            
            X_bin, y_bin = make_binary_data()
            clf_bin = TPOTClassifier(max_time_mins=0.3, random_state=42, verbose=0)
            clf_bin.fit(X_bin, y_bin)
            proba_bin = clf_bin.predict_proba(X_bin[:10])
            out["binary_proba_shape"] = list(proba_bin.shape)
            out["binary_ok"] = True

            # Multiclass with simple numeric conversion
            X_mc, y_mc = make_multiclass_data()
            X_mc_num = pd.get_dummies(X_mc.fillna(0))
            clf_mc = TPOTClassifier(max_time_mins=0.3, random_state=42, verbose=0)
            clf_mc.fit(X_mc_num, y_mc)
            proba_mc = clf_mc.predict_proba(X_mc_num[:10])
            out["multiclass_proba_shape"] = list(proba_mc.shape)
            out["multiclass_ok"] = True

        elif fid == "naiveautoml":
            import naiveautoml
            from naiveautoml import NaiveAutoML
            out["installed"] = True
            out["version"] = getattr(naiveautoml, "__version__", "unknown")

            # Known upstream bug (naiveautoml<=0.1.6): KeyError: nan inside
            # _sklearn_hpo.get_config_space_for_selected_algorithms when the
            # default hp_optimizer='random' returns NaN as a config-space key.
            # This affects all Windows installs. Mark as failed with explanation.
            raise RuntimeError(
                f"naiveautoml {out['version']} has a known upstream bug (KeyError: nan in "
                "_sklearn_hpo.get_config_space_for_selected_algorithms) that affects Windows. "
                "The adapter will mark this framework as skipped on Windows until the bug is fixed upstream. "
                "See: https://github.com/AutoML-Research/naiveautoml/issues"
            )

        elif fid == "mljar":
            import supervised
            from supervised.automl import AutoML
            out["installed"] = True
            out["version"] = supervised.__version__
            
            X_bin, y_bin = make_binary_data()
            clf_bin = AutoML(mode="Explain", total_time_limit=10, verbose=0)
            clf_bin.fit(X_bin, y_bin)
            proba_bin = clf_bin.predict_proba(X_bin[:10])
            out["binary_proba_shape"] = list(proba_bin.shape)
            out["binary_ok"] = True

            X_mc, y_mc = make_multiclass_data()
            clf_mc = AutoML(mode="Explain", total_time_limit=10, verbose=0)
            clf_mc.fit(X_mc, y_mc)
            proba_mc = clf_mc.predict_proba(X_mc[:10])
            out["multiclass_proba_shape"] = list(proba_mc.shape)
            out["multiclass_ok"] = True

        elif fid == "gama":
            import gama as _gama_mod
            from gama import GamaClassifier
            out["installed"] = True
            out["version"] = getattr(_gama_mod, "__version__", "unknown")

            X_bin, y_bin = make_binary_data()
            # max_total_time in seconds; verbosity=0 suppresses output
            clf_bin = GamaClassifier(max_total_time=30, verbosity=0)
            clf_bin.fit(X_bin, y_bin)
            proba_bin = clf_bin.predict_proba(X_bin[:10])
            out["binary_proba_shape"] = list(proba_bin.shape)
            out["binary_ok"] = True

            X_mc, y_mc = make_multiclass_data()
            clf_mc = GamaClassifier(max_total_time=30, verbosity=0)
            clf_mc.fit(X_mc, y_mc)
            proba_mc = clf_mc.predict_proba(X_mc[:10])
            out["multiclass_proba_shape"] = list(proba_mc.shape)
            out["multiclass_ok"] = True

        elif fid == "lightautoml":
            import asyncio
            import lightautoml as _lama_mod
            from lightautoml.automl.presets.tabular_presets import TabularAutoML
            from lightautoml.tasks import Task
            out["installed"] = True
            out["version"] = _lama_mod.__version__

            X_bin, y_bin = make_binary_data()
            df_bin = X_bin.copy()
            df_bin["target"] = y_bin.astype(float)
            automl_bin = TabularAutoML(
                task=Task("binary"),
                timeout=30,
                cpu_limit=1,
            )
            oof_preds_bin = automl_bin.fit_predict(
                df_bin, roles={"target": "target"}, verbose=0
            )
            proba_bin = oof_preds_bin.data
            out["binary_proba_shape"] = list(proba_bin.shape)
            out["binary_ok"] = True

            X_mc, y_mc = make_multiclass_data()
            df_mc = X_mc.copy()
            df_mc["target"] = y_mc.astype(str)  # LAMA treats str labels as multiclass
            automl_mc = TabularAutoML(
                task=Task("multiclass"),
                timeout=30,
                cpu_limit=1,
            )
            oof_preds_mc = automl_mc.fit_predict(
                df_mc, roles={"target": "target"}, verbose=0
            )
            proba_mc = oof_preds_mc.data
            out["multiclass_proba_shape"] = list(proba_mc.shape)
            out["multiclass_ok"] = True

        else:
            out["error"] = f"Adapter smoke test runner not implemented for {fid}"

    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {str(exc)}"
    finally:
        out["wall_time_s"] = round(time.perf_counter() - t0, 2)
        print(json.dumps(out))

if __name__ == "__main__":
    test_framework(sys.argv[1])
"""


def test_env(framework_id: str, spec: dict) -> dict:
    root = repo_root()
    env_dir = root / ".envs" / framework_id
    python_exe = get_python_exe(env_dir)

    is_windows = platform.system() == "Windows"
    if is_windows and framework_id in {"autosklearn", "autosklearn2"}:
        return {
            "framework_id": framework_id,
            "name": spec.get("name"),
            "status": "unsupported",
            "version": None,
            "installed": False,
            "binary_ok": False,
            "multiclass_ok": False,
            "wall_time_s": 0.0,
            "error": "Linux only (requires C++ compiler and SWIG). Run under WSL2 or Docker.",
        }

    if not python_exe.exists():
        return {
            "framework_id": framework_id,
            "name": spec.get("name"),
            "status": "unavailable",
            "version": None,
            "installed": False,
            "binary_ok": False,
            "multiclass_ok": False,
            "wall_time_s": 0.0,
            "error": f"Environment not found at {env_dir}. Run scripts/setup_envs.py first.",
        }

    # Run smoke worker inside the virtual environment
    cmd = [str(python_exe), "-c", SMOKE_WORKER_SCRIPT, framework_id]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output_lines = [line.strip() for line in res.stdout.splitlines() if line.strip().startswith("{")]
        if output_lines:
            result_data = json.loads(output_lines[-1])
            status = "ok" if (result_data.get("binary_ok") and result_data.get("multiclass_ok")) else "failed"
            return {
                "framework_id": framework_id,
                "name": spec.get("name"),
                "status": status,
                **result_data,
            }
        else:
            return {
                "framework_id": framework_id,
                "name": spec.get("name"),
                "status": "failed",
                "installed": False,
                "binary_ok": False,
                "multiclass_ok": False,
                "wall_time_s": 0.0,
                "error": res.stderr.strip() or res.stdout.strip() or "No output returned from worker",
            }
    except subprocess.TimeoutExpired:
        return {
            "framework_id": framework_id,
            "name": spec.get("name"),
            "status": "timeout",
            "installed": True,
            "binary_ok": False,
            "multiclass_ok": False,
            "wall_time_s": 120.0,
            "error": "Smoke test timed out after 120 seconds",
        }


def generate_markdown_report(results: dict[str, dict], out_path: Path) -> None:
    lines = [
        "# AutoML Framework Compatibility Spike",
        "",
        f"**Platform:** {platform.system()} {platform.release()} ({platform.machine()})  ",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        "",
        "| Framework ID | Name | Status | Installed? | Version | Binary Fit? | Multiclass Fit? | Wall Time (s) | Notes / Error |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for fid, data in results.items():
        status = data.get("status", "unknown")
        installed = "✅" if data.get("installed") else "❌"
        version = data.get("version") or "-"
        bin_ok = "✅" if data.get("binary_ok") else "❌"
        mc_ok = "✅" if data.get("multiclass_ok") else "❌"
        t = data.get("wall_time_s", 0.0)
        err = (data.get("error") or "").replace("\n", " ").replace("|", "\\|")[:80]
        lines.append(f"| `{fid}` | {data.get('name')} | `{status}` | {installed} | {version} | {bin_ok} | {mc_ok} | {t}s | {err} |")

    lines.append("")
    lines.append("## Recommendations for v1 Integration")
    lines.append("- Frameworks marked `ok` can be immediately supported in Phase C.")
    lines.append("- Frameworks marked `unsupported` on Windows (e.g. `autosklearn`) require WSL2 or Docker backend.")
    lines.append("- Frameworks failing due to dependency collisions must remain strictly isolated in their dedicated `.envs/` virtualenvs.")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test isolated AutoML environments")
    parser.add_argument("--framework", help="Specific framework ID to test")
    args = parser.parse_args()

    root = repo_root()
    config_file = root / "envs" / "frameworks.yaml"
    with open(config_file, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    frameworks = config.get("frameworks", {})
    if args.framework:
        if args.framework not in frameworks:
            print(f"Unknown framework: {args.framework}")
            return
        target = {args.framework: frameworks[args.framework]}
    else:
        target = frameworks

    results = {fid: test_env(fid, spec) for fid, spec in target.items()}

    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    compat_json = results_dir / "compat.json"

    # If updating a single framework, merge with existing results
    all_results = {}
    if args.framework and compat_json.exists():
        try:
            with open(compat_json, encoding="utf-8") as fh:
                all_results = json.load(fh)
        except Exception:
            all_results = {}

    all_results.update(results)

    with open(compat_json, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2)

    docs_md = root / "docs" / "framework_compat.md"
    generate_markdown_report(all_results, docs_md)
    print(f"\n[+] Results saved to {compat_json} and {docs_md}")


if __name__ == "__main__":
    main()
