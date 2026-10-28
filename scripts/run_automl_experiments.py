"""Automated Batch Experiment Runner for AutoML Benchmarking.

Runs the two-tier benchmark (Classic ML baselines + AutoML contenders) across
registry datasets following docs/protocol.md.

Features:
  - Resumable execution: skips already completed (dataset, framework, budget) runs.
  - Progress logging & failure diagnostics.
  - Estimates total runtime before starting.
  - Persists all results in SQLite/PostgreSQL DB and exports consolidated CSV.

Usage:
  python scripts/run_automl_experiments.py --budget 60 --tier quick
  python scripts/run_automl_experiments.py --budget 300 --datasets breast_cancer,wine,adult
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

import pandas as pd
import yaml

from acbs.benchmark.aggregate import results_to_dataframe
from acbs.benchmark.engine import BenchmarkConfig, estimate_cost, run_benchmark
from acbs.data.loader import load_dataset
from acbs.storage import Repository, make_session_factory
from acbs.storage.models import ExperimentRow
from acbs.utils.paths import repo_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("acbs.experiment_runner")


def _load_available_datasets(dataset_filter: list[str] | None = None) -> list[dict]:
    reg_path = repo_root() / "data" / "registry.yaml"
    if not reg_path.exists():
        logger.error(f"Registry not found: {reg_path}")
        return []

    with open(reg_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    all_ds = data.get("datasets", [])
    # Filter out kaggle by default unless requested
    available = []
    for d in all_ds:
        if d.get("source") == "kaggle" and (not dataset_filter or d["name"] not in dataset_filter):
            continue
        if dataset_filter and d["name"] not in dataset_filter:
            continue
        available.append(d)

    return available


def _get_completed_keys(session) -> set[tuple[str, str, int]]:
    """Return set of (dataset_name, framework/model, budget_s) already completed."""
    completed = set()
    try:
        exps = session.query(ExperimentRow).filter(ExperimentRow.status == "completed").all()
        for exp in exps:
            for run in exp.model_runs:
                if run.status == "ok":
                    key = (exp.dataset.name, run.framework or run.model_name, int(run.budget_s or 0))
                    completed.add(key)
    except Exception as exc:
        logger.warning(f"Could not load completed experiments from DB: {exc}")
    return completed


def main() -> int:
    parser = argparse.ArgumentParser(description="Run batch AutoML benchmark experiments.")
    parser.add_argument("--budget", type=int, default=60, help="Per-framework budget in seconds (default: 60)")
    parser.add_argument("--frameworks", type=str, default="autogluon,flaml,gama,h2o,lightautoml,mljar,tpot",
                        help="Comma-separated AutoML framework IDs")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset names (default: all registry datasets except kaggle)")
    parser.add_argument("--outer-folds", type=int, default=1,
                        help="Outer holdout/folds (1 = single 80/20 holdout, 3 = repeated)")
    parser.add_argument("--n-jobs", type=int, default=4, help="CPU threads per framework")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip already completed runs")
    parser.add_argument("--dry-run", action="store_true", help="Print runtime estimate and dataset plan only")
    parser.add_argument("--output-csv", type=str, default="results/automl_benchmark_results.csv",
                        help="Output path for consolidated CSV")

    args = parser.parse_args()

    fw_list = [f.strip() for f in args.frameworks.split(",") if f.strip()]
    ds_filter = [d.strip() for d in args.datasets.split(",") if d.strip()] if args.datasets else None

    datasets = _load_available_datasets(ds_filter)
    if not datasets:
        logger.error("No matching datasets found to benchmark.")
        return 1

    logger.info(f"Target datasets ({len(datasets)}): {[d['name'] for d in datasets]}")
    logger.info(f"Target AutoML frameworks ({len(fw_list)}): {fw_list}")
    logger.info(f"Budget per framework: {args.budget}s | Outer folds: {args.outer_folds} | n_jobs: {args.n_jobs}")

    # Cost estimation
    cfg_for_est = BenchmarkConfig(
        models=["all"],
        automl_frameworks=fw_list,
        automl_budget_s=args.budget,
        outer_folds=args.outer_folds,
    )
    est = estimate_cost(cfg_for_est, n_datasets=len(datasets))
    total_min = est.get("total_minutes") or (est.get("total_seconds", 0) / 60.0)
    logger.info(f"Estimated total wall time: {total_min:.1f} minutes (~{total_min / 60.0:.2f} hours)")

    if args.dry_run:
        logger.info("Dry-run complete. Exiting without execution.")
        return 0

    results_dir = repo_root() / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    all_runs_data: list[dict] = []

    # Check already collected CSV data if resuming
    csv_path = repo_root() / args.output_csv
    if args.resume and csv_path.exists():
        try:
            existing_df = pd.read_csv(csv_path)
            all_runs_data.extend(existing_df.to_dict(orient="records"))
            logger.info(f"Loaded {len(existing_df)} existing records from {csv_path}")
        except Exception as exc:
            logger.warning(f"Could not read existing CSV: {exc}")

    SessionLocal = make_session_factory()
    with SessionLocal() as session:
        repo = Repository(session)
        completed_keys = _get_completed_keys(session) if args.resume else set()

        for idx, ds_meta in enumerate(datasets, 1):
            ds_name = ds_meta["name"]
            logger.info(f"\n[{idx}/{len(datasets)}] Processing dataset: {ds_name}...")

            try:
                ds = load_dataset(ds_name)
            except Exception as exc:
                logger.error(f"Failed to load dataset {ds_name}: {exc}")
                continue

            # Determine frameworks still needing run
            pending_fws = [
                fw for fw in fw_list
                if (ds_name, fw, args.budget) not in completed_keys
            ]

            if not pending_fws and (ds_name, "random_forest", 0) in completed_keys:
                logger.info(f"Dataset {ds_name} already completed in previous runs. Skipping.")
                continue

            config = BenchmarkConfig(
                models=["all"],
                automl_frameworks=pending_fws,
                automl_budget_s=args.budget,
                automl_n_jobs=args.n_jobs,
                outer_folds=args.outer_folds,
                seed=42,
            )

            exp_record = repo.create_experiment(
                dataset_id=ds.id,
                config=config.model_dump(),
                preferences={"preset": "balanced"},
                seed=config.seed,
            )

            t0 = time.perf_counter()
            try:
                result = run_benchmark(
                    dataset=ds,
                    cfg=config.model_dump(),
                    progress_callback=lambda name, step, total, message, status=None, elapsed_s=None: logger.info(
                        "  [%3d%%] %s [%s]",
                        int(100 * step / max(total, 1)),
                        message,
                        status or "running",
                    ),
                )
                elapsed = time.perf_counter() - t0
                repo.save_results(exp_record.id, result.model_dump())
                results_df = results_to_dataframe(result)
                logger.info(f"Dataset {ds_name} benchmark finished in {elapsed:.1f}s.")

                # Append to consolidated results
                for _, row in results_df.iterrows():
                    rec = row.to_dict()
                    rec["dataset_name"] = ds_name
                    rec["domain"] = ds_meta.get("domain", "general")
                    rec["task"] = ds_meta.get("task", "binary")
                    all_runs_data.append(rec)

                # Incremental CSV save
                out_df = pd.DataFrame(all_runs_data)
                out_df.to_csv(csv_path, index=False)
                logger.info(f"Updated {csv_path} ({len(out_df)} total records).")

            except Exception as exc:
                logger.error(f"Experiment failed for dataset {ds_name}: {exc}", exc_info=True)
                repo.update_status(exp_record.id, "failed", error=str(exc))

    logger.info("\n==========================================")
    logger.info(f"All benchmark runs completed! Consolidated results: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
