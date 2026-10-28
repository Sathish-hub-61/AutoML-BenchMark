from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from acbs.recommend.scoring import UserPreferences, automl_value_analysis, score_models

_FRAMEWORK_STRATEGIES = {
    "autogluon": "stacked ensembles of preset pipelines",
    "autosklearn": "Bayesian optimisation of scikit-learn pipelines",
    "autosklearn2": "Bayesian optimisation of iterative algorithms",
    "flaml": "cost-frugal optimisation",
    "gama": "evolutionary optimisation of scikit-learn pipelines",
    "h2o": "random search and ensembling",
    "lightautoml": "Bayesian optimisation of linear models and GBMs",
    "mljar": "custom data-science pipelines with stacked ensembling",
    "naiveautoml": "sequential algorithm selection with hyperparameter optimisation",
    "tpot": "evolutionary optimisation of scikit-learn pipelines",
}


def _get_framework_info(framework_id: str) -> dict[str, str]:
    root = Path(__file__).resolve().parent.parent.parent.parent
    info_path = root / "configs" / "frameworks_info.yaml"
    if info_path.exists():
        try:
            with open(info_path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
                fws = data.get("frameworks", {})
                if framework_id in fws:
                    return fws[framework_id]
        except Exception:
            pass
    return {
        "name": framework_id.title(),
        "strategy": _FRAMEWORK_STRATEGIES.get(
            framework_id.lower(), "automated machine learning pipeline search"
        ),
    }


def generate_explanation(
    rec: pd.Series,
    results_df: pd.DataFrame,
    profile: dict[str, Any],
    prefs: UserPreferences,
    stability: dict[str, float] | None = None,
) -> dict[str, Any]:
    name = str(rec["model_name"])
    kind = str(rec.get("kind", "classic"))
    fw_id = str(rec.get("framework") or name).lower()
    is_automl = kind == "automl" or fw_id in _FRAMEWORK_STRATEGIES

    scored = score_models(results_df, prefs)
    eligible = scored[~scored["excluded"]].sort_values("rank")
    runner = eligible.iloc[1] if len(eligible) > 1 else None
    f1 = rec.get("f1_macro")
    acc = rec.get("accuracy")
    ttrain = rec.get("training_time")
    tpred = rec.get("prediction_time_per_1k")
    stab = (stability or {}).get(name, 0.0) * 100

    val = automl_value_analysis(results_df)

    display_name = name.replace("_", " ").title()
    sentences = [
        f"{display_name} is recommended under the current priorities.",
    ]

    if is_automl:
        fw_info = _get_framework_info(fw_id)
        strategy_str = fw_info.get(
            "strategy", _FRAMEWORK_STRATEGIES.get(fw_id, "automated pipeline search")
        )
        sentences.append(f"{display_name} employs {strategy_str}.")

    if pd.notna(f1) and pd.notna(acc):
        sentences.append(f"Hold-out F1-macro is {f1:.3f} and accuracy is {acc:.3f}.")
    else:
        sentences.append("Hold-out predictive metrics were computed on the test split.")

    if runner is not None:
        runner_name = str(runner["model_name"]).replace("_", " ").title()
        sentences.append(
            f"The runner-up is {runner_name} (score {runner['composite_score']:.3f} vs {rec['composite_score']:.3f})."
        )

    if pd.notna(ttrain) and pd.notna(tpred):
        sentences.append(
            f"It trained in {ttrain:.2f}s and predicts in {tpred * 1000:.2f} ms per 1,000 rows."
        )

    if profile.get("is_imbalanced"):
        sentences.append(
            f"The dataset is imbalanced (ratio {profile.get('imbalance_ratio', 0):.1f}:1); ranking emphasised F1 and recall rather than raw accuracy."
        )

    if val["has_both_tiers"]:
        if is_automl:
            sentences.append(f"AutoML Value Analysis: {val['summary']}")
        else:
            sentences.append(
                f"AutoML Comparison: Plain {display_name} is recommended over AutoML contenders ({val['summary']})."
            )

    if val.get("is_statistical_tie"):
        sentences.append(
            "Statistical note: The top AutoML framework and classic baseline are within statistical noise."
        )

    sentences.append(f"Recommendation stability under ±20% weight noise is {stab:.0f}%.")

    why_not = []
    others = eligible[eligible["model_name"] != name].head(2)
    for _, row in others.iterrows():
        other_kind = f" [{row['kind']}]" if "kind" in row and pd.notna(row["kind"]) else ""
        why_not.append(
            f"{row['model_name']}{other_kind}: lower composite score ({row['composite_score']:.3f}); "
            f"F1={row.get('f1_macro', float('nan')):.3f}."
        )

    caveats = list(profile.get("warnings") or [])
    if is_automl:
        budget_ratio = rec.get("budget_used_ratio")
        if pd.notna(budget_ratio) and float(budget_ratio) > 0:
            caveats.append(
                f"Framework utilized {float(budget_ratio) * 100:.1f}% of its allocated time budget."
            )
        if pd.notna(tpred) and float(tpred) * 1000 > 50:
            caveats.append(
                f"Inference latency is high ({float(tpred) * 1000:.1f} ms/1k rows) due to model ensembling."
            )
        caveats.append(
            "AutoML search may be non-deterministic; fixed seeds are used where supported."
        )

    return {
        "headline": f"Recommended: {display_name}",
        "model_name": name,
        "paragraphs": sentences,
        "why_not": why_not,
        "caveats": caveats,
        "stability_pct": stab,
        "score": float(rec.get("composite_score") or 0),
        "automl_value": val,
    }
