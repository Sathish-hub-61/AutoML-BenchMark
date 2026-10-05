from __future__ import annotations

from acbs.data.profiler import DatasetProfile


def get_imbalance_strategy(
    profile: DatasetProfile,
    config: dict,
    override: str | None = None,
) -> str:
    if override and override != "auto":
        return override
    if not profile.is_imbalanced:
        return "none"
    minority = min(profile.class_counts.values()) if profile.class_counts else 0
    if profile.n_rows > int(config.get("smote_max_rows", 20_000)):
        return "class_weight"
    if minority >= int(config.get("smote_min_minority", 6)) and config.get("allow_smote"):
        return "smote"
    return "class_weight"
