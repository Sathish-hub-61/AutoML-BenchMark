from __future__ import annotations

from typing import Any

import pandas as pd

from acbs.recommend.scoring import UserPreferences, score_models


def generate_explanation(
    rec: pd.Series,
    results_df: pd.DataFrame,
    profile: dict[str, Any],
    prefs: UserPreferences,
    stability: dict[str, float] | None = None,
) -> dict[str, Any]:
    name = rec["model_name"]
    scored = score_models(results_df, prefs)
    eligible = scored[~scored["excluded"]].sort_values("rank")
    runner = eligible.iloc[1] if len(eligible) > 1 else None
    f1 = rec.get("f1_macro")
    acc = rec.get("accuracy")
    ttrain = rec.get("training_time")
    tpred = rec.get("prediction_time_per_1k")
    stab = (stability or {}).get(name, 0.0) * 100
    sentences = [
        f"{name.replace('_', ' ').title()} is recommended under the current priorities.",
        f"Hold-out F1-macro is {f1:.3f} and accuracy is {acc:.3f}."
        if pd.notna(f1) and pd.notna(acc)
        else "Hold-out predictive metrics were computed on the test split.",
    ]
    if runner is not None:
        sentences.append(
            f"The runner-up is {runner['model_name']} (score {runner['composite_score']:.3f} vs {rec['composite_score']:.3f})."
        )
    if pd.notna(ttrain) and pd.notna(tpred):
        sentences.append(
            f"It trained in {ttrain:.2f}s and predicts in {tpred * 1000:.2f} ms per 1,000 rows."
        )
    if profile.get("is_imbalanced"):
        sentences.append(
            f"The dataset is imbalanced (ratio {profile.get('imbalance_ratio', 0):.1f}:1); ranking emphasised F1 and recall rather than raw accuracy."
        )
    sentences.append(f"Recommendation stability under ±20% weight noise is {stab:.0f}%.")
    why_not = []
    others = eligible[eligible["model_name"] != name].head(2)
    for _, row in others.iterrows():
        why_not.append(
            f"{row['model_name']}: lower composite score ({row['composite_score']:.3f}); "
            f"F1={row.get('f1_macro', float('nan')):.3f}."
        )
    caveats = list(profile.get("warnings") or [])
    return {
        "headline": f"Recommended: {name}",
        "model_name": name,
        "paragraphs": sentences,
        "why_not": why_not,
        "caveats": caveats,
        "stability_pct": stab,
        "score": float(rec.get("composite_score") or 0),
    }
