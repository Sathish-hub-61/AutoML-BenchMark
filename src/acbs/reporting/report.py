from __future__ import annotations

from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from acbs.benchmark.aggregate import results_to_dataframe
from acbs.benchmark.engine import BenchmarkResult
from acbs.recommend.explain import generate_explanation
from acbs.recommend.scoring import UserPreferences, pareto_front, score_models, sensitivity_analysis
from acbs.reporting.charts import (
    accuracy_vs_time_scatter,
    grouped_metrics_bar,
    performance_bar,
    score_breakdown_stacked_bar,
    time_comparison,
)


def _env() -> Environment:
    folder = Path(__file__).parent / "templates"
    return Environment(loader=FileSystemLoader(str(folder)), autoescape=select_autoescape(["html"]))


def build_html_report(result: BenchmarkResult, prefs: UserPreferences | None = None) -> str:
    prefs = prefs or UserPreferences()
    df = results_to_dataframe(result)
    scored = score_models(df, prefs)
    eligible = scored[~scored["excluded"]]
    rec_row = eligible.sort_values("rank").iloc[0] if len(eligible) else scored.iloc[0]
    stability = sensitivity_analysis(df, prefs)
    explanation = generate_explanation(rec_row, df, result.profile, prefs, stability)
    pareto = pareto_front(df)
    figs = [
        performance_bar(df).to_html(full_html=False, include_plotlyjs="cdn"),
        grouped_metrics_bar(df).to_html(full_html=False, include_plotlyjs=False),
        time_comparison(df).to_html(full_html=False, include_plotlyjs=False),
        accuracy_vs_time_scatter(df, pareto).to_html(full_html=False, include_plotlyjs=False),
        score_breakdown_stacked_bar(scored).to_html(full_html=False, include_plotlyjs=False),
    ]
    table = scored.drop(columns=["best_params"], errors="ignore").to_html(index=False)
    failed = [m.model_dump() for m in result.models if m.status != "ok"]
    tmpl = _env().get_template("report.html.j2")
    return tmpl.render(
        recommended=explanation["model_name"],
        explanation=explanation,
        profile=result.profile,
        table_html=table,
        figures=figs,
        failed=failed,
        seed=result.seed,
        versions=result.lib_versions,
    )


def export_csv(df: pd.DataFrame) -> str:
    return df.to_csv(index=False)


def export_pdf(html: str) -> bytes | None:
    try:
        from io import BytesIO

        from xhtml2pdf import pisa

        out = BytesIO()
        pisa.CreatePDF(html, dest=out)
        return out.getvalue()
    except Exception:
        return None
