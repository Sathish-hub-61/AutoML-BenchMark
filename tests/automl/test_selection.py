"""Tests for automatic framework selection (acbs.automl.selection)."""

from __future__ import annotations

from acbs.automl.selection import load_policy, select_frameworks

BASE_POLICY = {
    "budget_by_size": {"small": 60, "medium": 120, "large": 300},
    "default_budget_s": 60,
    "total_wall_budget_s": 900,
    "wall_clock_multiplier": 1.25,
    "max_frameworks": 4,
    "frameworks": {},
}


def _fw(
    fid: str, *, kind: str = "single_pipeline", installed: bool = True, os_supported: bool = True
) -> dict:
    return {
        "id": fid,
        "name": fid.title(),
        "strategy": "test strategy",
        "kind": kind,
        "installed": installed,
        "version": "1.0",
        "os_supported": os_supported,
        "reason": None,
    }


PROFILE = {"n_rows": 600, "size_category": "small"}


def test_selects_only_installed_and_supported():
    avail = [
        _fw("flaml"),
        _fw("gama", installed=False),
        _fw("autosklearn", os_supported=False),
    ]
    res = select_frameworks(PROFILE, availability=avail, policy=BASE_POLICY)

    assert res.frameworks == ["flaml"]
    assert "not installed" in res.skipped["gama"]
    assert "unsupported" in res.skipped["autosklearn"]
    # every decision carries a human-readable reason
    assert res.selected["flaml"]


def test_budget_sized_by_dataset_when_not_overridden():
    avail = [_fw("flaml")]
    small = select_frameworks(PROFILE, availability=avail, policy=BASE_POLICY)
    large = select_frameworks(
        {"n_rows": 500_000, "size_category": "large"}, availability=avail, policy=BASE_POLICY
    )
    override = select_frameworks(PROFILE, budget_s=300, availability=avail, policy=BASE_POLICY)

    assert small.budget_s == 60
    assert large.budget_s == 300
    assert override.budget_s == 300


def test_wall_budget_caps_framework_count():
    avail = [_fw(f"fw{i}") for i in range(6)]
    policy = {**BASE_POLICY, "total_wall_budget_s": 100}  # 100 / 75 -> one slot
    res = select_frameworks(PROFILE, availability=avail, policy=policy)

    assert len(res.frameworks) == 1


def test_max_frameworks_cap():
    avail = [_fw(f"fw{i}") for i in range(10)]
    policy = {**BASE_POLICY, "max_frameworks": 2}
    res = select_frameworks(PROFILE, availability=avail, policy=policy)

    assert len(res.frameworks) == 2


def test_prefers_search_strategy_diversity():
    avail = [
        _fw("flaml", kind="single_pipeline"),
        _fw("autogluon", kind="stacked_ensemble"),
        _fw("h2o", kind="stacked_ensemble"),
        _fw("tpot", kind="single_pipeline"),
    ]
    policy = {**BASE_POLICY, "max_frameworks": 2}
    res = select_frameworks(PROFILE, availability=avail, policy=policy)

    kinds = {"flaml": "single_pipeline", "autogluon": "stacked_ensemble", "h2o": "stacked_ensemble"}
    chosen_kinds = {kinds[fid] for fid in res.frameworks}
    assert len(chosen_kinds) == 2  # one representative per strategy family
    assert res.frameworks[0] == "flaml"  # priority order within the picks


def test_nothing_available_returns_empty_without_crash():
    res = select_frameworks(
        PROFILE, availability=[_fw("gama", installed=False)], policy=BASE_POLICY
    )

    assert res.frameworks == []
    assert "gama" in res.skipped


def test_missing_profile_defaults_to_small_budget():
    res = select_frameworks(None, availability=[_fw("flaml")], policy=BASE_POLICY)
    assert res.budget_s == 60


def test_real_policy_skips_frameworks_too_large_for_dataset():
    policy = load_policy()
    assert "tpot" in policy["frameworks"]
    avail = [_fw("flaml"), _fw("tpot")]
    res = select_frameworks(
        {"n_rows": 200_000, "size_category": "large"}, availability=avail, policy=policy
    )

    assert "flaml" in res.frameworks
    assert "tpot" not in res.frameworks
    assert "too large" in res.skipped["tpot"]


def test_as_dict_is_json_friendly():
    res = select_frameworks(PROFILE, availability=[_fw("flaml")], policy=BASE_POLICY)
    d = res.as_dict()
    assert d["frameworks"] == ["flaml"]
    assert d["budget_s"] == 60
    assert "flaml" in d["selected"]
