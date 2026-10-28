from __future__ import annotations

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from acbs import __version__
from acbs.benchmark.aggregate import results_to_dataframe
from acbs.benchmark.engine import run_benchmark
from acbs.data.loader import load_dataset
from acbs.recommend.explain import generate_explanation
from acbs.recommend.scoring import UserPreferences, score_models, sensitivity_analysis

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def version() -> None:
    console.print(__version__)


@app.command()
def run(
    dataset: str = typer.Option(..., "--dataset"),
    target: str | None = typer.Option(None, "--target"),
    models: str = typer.Option("all", "--models"),
    automl: str | None = typer.Option(
        None, "--automl", help="Comma-separated AutoML framework IDs"
    ),
    budget: int = typer.Option(60, "--budget", help="Time budget per AutoML framework in seconds"),
    tuning: str = typer.Option("none", "--tuning"),
    cv_folds: int = typer.Option(5, "--cv-folds"),
    n_trials: int = typer.Option(10, "--n-trials"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    loaded = load_dataset(dataset, target)
    model_list = (
        []
        if models == "none"
        else (["all"] if models == "all" else [m.strip() for m in models.split(",")])
    )
    automl_list = [f.strip() for f in automl.split(",")] if automl else []
    result = run_benchmark(
        loaded,
        {
            "models": model_list,
            "automl_frameworks": automl_list,
            "automl_budget_s": budget,
            "tuning": tuning,
            "cv_folds": cv_folds,
            "n_trials": n_trials,
            "seed": seed,
        },
    )
    df = results_to_dataframe(result)
    prefs = UserPreferences()
    scored = score_models(df, prefs)
    table = Table(title=f"Benchmark: {dataset}")
    cols = [
        "rank",
        "model_name",
        "kind",
        "status",
        "accuracy",
        "f1_macro",
        "roc_auc",
        "training_time",
        "composite_score",
    ]
    for col in cols:
        table.add_column(col)
    view = scored.sort_values(["excluded", "rank"], na_position="last")
    for _, row in view.iterrows():
        table.add_row(
            *[str(row.get(c, "") if pd.notna(row.get(c, float("nan"))) else "") for c in cols]
        )
    console.print(table)
    eligible = scored[~scored["excluded"]]
    if not eligible.empty:
        rec = eligible.sort_values("rank").iloc[0]
        exp = generate_explanation(rec, df, result.profile, prefs, sensitivity_analysis(df, prefs))
        console.print(f"\n[bold green]{exp['headline']}[/bold green]")
        for p in exp["paragraphs"]:
            console.print(p)
