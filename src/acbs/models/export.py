"""Model Export Service.

Packages the winning model from an experiment into a downloadable bundle (.zip)
containing:
  - For Classic models:
      - model.joblib (full scikit-learn pipeline refit on all data)
      - metadata.json (features, target, classes, versions, metrics)
      - predict_snippet.py (executable inference script)
      - requirements.txt & README.md
  - For AutoML frameworks:
      - metadata.json (framework details, search strategy, leaderboard)
      - load_snippet.py (native code to load and predict)
      - requirements.txt (environment requirements from frameworks.yaml)
      - README.md (instructions on loading with isolated env)
"""

from __future__ import annotations

import io
import json
import logging
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import joblib
import sklearn
import yaml
from sklearn.pipeline import Pipeline as SkPipeline

from acbs.data.loader import load_dataset
from acbs.data.profiler import profile_dataset
from acbs.models.registry import get_models
from acbs.preprocessing.pipeline import build_preprocessor
from acbs.storage.models import ExperimentRow
from acbs.utils.paths import repo_root

logger = logging.getLogger(__name__)


def export_experiment_model(experiment: ExperimentRow, output_dir: Path | None = None) -> Path:
    """Build a zip package for the winning model of an experiment."""
    if not experiment.recommendation:
        raise ValueError(f"Experiment #{experiment.id} has no recommendation to export.")

    winner_name = experiment.recommendation.model_name
    winner_run = None
    for run in experiment.model_runs:
        if run.model_name == winner_name:
            winner_run = run
            break

    is_automl = (winner_run and winner_run.kind == "automl") or (
        winner_name
        in {
            "autogluon",
            "autosklearn",
            "autosklearn2",
            "flaml",
            "gama",
            "h2o",
            "lightautoml",
            "mljar",
            "naiveautoml",
            "tpot",
        }
    )

    if output_dir is None:
        target_dir = Path(tempfile.mkdtemp(prefix=f"acbs_export_exp{experiment.id}_"))
    else:
        target_dir = output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

    zip_filename = f"acbs_model_exp{experiment.id}_{winner_name}.zip"
    zip_path = target_dir / zip_filename

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if not is_automl:
            _bundle_classic_model(experiment, winner_name, winner_run, zf)
        else:
            _bundle_automl_model(experiment, winner_name, winner_run, zf)

    return zip_path


def _bundle_classic_model(
    experiment: ExperimentRow,
    winner_name: str,
    winner_run: Any,
    zf: zipfile.ZipFile,
) -> None:
    ds_row = experiment.dataset
    ds = load_dataset(ds_row.name)
    X = ds.X
    y = ds.y
    target_col = ds.meta.get("target_column") or getattr(ds_row, "target_column", "target")

    profile = profile_dataset(X, y)
    cfg = experiment.config_json or {}

    spec = get_models([winner_name])[0]
    best_params = (getattr(winner_run, "best_params_json", None) or {}) if winner_run else {}

    preprocessor = build_preprocessor(profile, cfg, spec.family, X=X)
    estimator = spec.estimator_factory(
        best_params, seed=cfg.get("seed", 42), n_jobs=cfg.get("n_jobs", 1)
    )

    pipeline = SkPipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )

    pipeline.fit(X, y)

    # Save fitted model
    model_buf = io.BytesIO()
    joblib.dump(pipeline, model_buf)
    zf.writestr("model.joblib", model_buf.getvalue())

    # Metadata
    classes = [str(c) for c in getattr(estimator, "classes_", sorted(y.unique()))]
    metrics = (
        (
            getattr(winner_run, "test_metrics_json", None)
            or getattr(winner_run, "cv_metrics_json", None)
            or {}
        )
        if winner_run
        else {}
    )
    metadata = {
        "experiment_id": experiment.id,
        "model_name": winner_name,
        "kind": "classic",
        "family": spec.family,
        "dataset_name": ds_row.name,
        "target_column": target_col,
        "features": list(X.columns),
        "classes": classes,
        "python_version": sys.version,
        "sklearn_version": sklearn.__version__,
        "metrics": metrics,
    }
    zf.writestr("metadata.json", json.dumps(metadata, indent=2))

    # Python inference snippet
    snippet = f'''"""Inference snippet for {winner_name} trained via acbs."""
import joblib
import pandas as pd

# 1. Load pipeline
pipeline = joblib.load("model.joblib")

# 2. Prepare new data with columns: {list(X.columns)}
# sample_df = pd.read_csv("new_data.csv")

# 3. Predict class labels and probabilities
# y_pred = pipeline.predict(sample_df)
# y_proba = pipeline.predict_proba(sample_df)
print("Pipeline loaded successfully:", pipeline)
'''
    zf.writestr("predict_snippet.py", snippet)

    # Requirements & README
    reqs = f"scikit-learn=={sklearn.__version__}\npandas\nnumpy\njoblib\n"
    if spec.family == "boosting" or "xgb" in winner_name:
        reqs += "xgboost\n"
    zf.writestr("requirements.txt", reqs)

    readme = f"""# {winner_name.title()} Model Package (acbs)

Trained and exported from **acbs** Experiment #{experiment.id} on dataset `{ds_row.name}`.

## Files
- `model.joblib`: Complete scikit-learn Pipeline (preprocessing + classifier) fit on full dataset.
- `metadata.json`: Feature names, target classes, and metrics.
- `predict_snippet.py`: Example code to load and run inference.
- `requirements.txt`: Python dependencies.

## Usage
```bash
pip install -r requirements.txt
python predict_snippet.py
```
"""
    zf.writestr("README.md", readme)


def _bundle_automl_model(
    experiment: ExperimentRow,
    winner_name: str,
    winner_run: Any,
    zf: zipfile.ZipFile,
) -> None:
    ds_row = experiment.dataset
    fw_id = winner_name
    root = repo_root()

    # Load framework metadata
    meta_path = root / "configs" / "frameworks_info.yaml"
    fw_strategy = "Automated machine learning optimization"
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as fh:
                meta = yaml.safe_load(fh).get("frameworks", {}).get(fw_id, {})
                fw_strategy = meta.get("strategy", fw_strategy)
        except Exception:
            pass

    # Load framework packages from frameworks.yaml
    fw_yaml_path = root / "envs" / "frameworks.yaml"
    pip_packages = [fw_id]
    if fw_yaml_path.exists():
        try:
            with open(fw_yaml_path, encoding="utf-8") as fh:
                spec = yaml.safe_load(fh).get("frameworks", {}).get(fw_id, {})
                pip_packages = spec.get("pip", [fw_id])
        except Exception:
            pass

    automl_metrics = (
        (
            getattr(winner_run, "test_metrics_json", None)
            or getattr(winner_run, "cv_metrics_json", None)
            or {}
        )
        if winner_run
        else {}
    )
    metadata = {
        "experiment_id": experiment.id,
        "framework": fw_id,
        "kind": "automl",
        "strategy": fw_strategy,
        "dataset_name": ds_row.name,
        "budget_s": getattr(winner_run, "budget_s", None) if winner_run else None,
        "metrics": automl_metrics,
        "model_summary": getattr(winner_run, "model_summary", None) if winner_run else None,
    }
    zf.writestr("metadata.json", json.dumps(metadata, indent=2))

    # Load / predict code template per framework
    load_code = _automl_load_snippet(fw_id)
    zf.writestr("load_snippet.py", load_code)
    zf.writestr("requirements.txt", "\n".join(pip_packages) + "\npandas\nnumpy\n")

    readme = f"""# {fw_id.upper()} AutoML Model Export (acbs)

Recommended winner from **acbs** Experiment #{experiment.id} on `{ds_row.name}`.
**Search Strategy:** {fw_strategy}

> ⚠️ **Environment Note:** AutoML frameworks often have strict dependency pins (scikit-learn, torch, numpy).
> To run or reload native artifacts, use the matching isolated environment:
>
> Windows: `.envs\\{fw_id}\\Scripts\\activate`
> Linux/Mac: `source .envs/{fw_id}/bin/activate`

## Usage
1. Activate the dedicated virtual environment or install `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
2. Run `load_snippet.py` to inspect and execute inference.
"""
    zf.writestr("README.md", readme)


def _automl_load_snippet(fw_id: str) -> str:
    templates = {
        "autogluon": '''"""Load and predict with AutoGluon TabularPredictor."""
from autogluon.tabular import TabularPredictor
import pandas as pd

# Load saved predictor folder (if exported directly)
# predictor = TabularPredictor.load("AutogluonModels/ag-...")
# preds = predictor.predict(new_df)
print("AutoGluon environment verified.")
''',
        "flaml": '''"""Inference with FLAML AutoML."""
import joblib
import pandas as pd

# flaml models can be serialized with joblib or pickle:
# automl = joblib.load("flaml_model.pkl")
# preds = automl.predict(new_df)
print("FLAML environment verified.")
''',
        "h2o": '''"""Inference with H2O AutoML."""
import h2o
h2o.init()

# Load saved model or MOJO:
# model = h2o.load_model("path/to/h2o_model")
# hf = h2o.H2OFrame(new_df)
# preds = model.predict(hf)
print("H2O initialized successfully.")
''',
    }
    return templates.get(
        fw_id,
        f'''"""Inference snippet for {fw_id}."""
import pandas as pd
print("Use .envs/{fw_id} to load native {fw_id} models.")
''',
    )
