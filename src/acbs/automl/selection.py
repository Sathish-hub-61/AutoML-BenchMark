"""Automatic AutoML framework selection — the system chooses, no human does.

Policy lives in ``configs/automl_selection.yaml``. Selection is deterministic:

1. Drop frameworks that are not installed or unsupported on this OS.
2. Drop frameworks whose dataset constraints (``min_rows`` / ``max_rows``)
   exclude the current dataset.
3. Rank the rest by policy ``priority``.
4. Compute how many frameworks fit the wall-clock budget.
5. Fill those slots preferring search-strategy diversity (one framework per
   ``kind`` first), then priority order.

Every decision — selected or skipped — carries a human-readable reason that is
stored with the experiment, so an automatic choice is always explainable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from acbs.automl.availability import framework_availability
from acbs.utils.logging import get_logger
from acbs.utils.paths import load_yaml, repo_root

logger = get_logger(__name__)

_POLICY_FILE = "automl_selection.yaml"


def load_policy() -> dict:
    """Load the selection policy from ``configs/automl_selection.yaml``."""
    return load_yaml(repo_root() / "configs" / _POLICY_FILE)


@dataclass
class SelectionResult:
    """Outcome of an automatic framework selection."""

    frameworks: list[str]
    budget_s: int
    selected: dict[str, str]
    skipped: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "frameworks": list(self.frameworks),
            "budget_s": self.budget_s,
            "selected": dict(self.selected),
            "skipped": dict(self.skipped),
        }


def select_frameworks(
    profile: dict[str, Any] | None = None,
    *,
    budget_s: int | None = None,
    availability: list[dict] | None = None,
    policy: dict[str, Any] | None = None,
    wall_budget_s: int | None = None,
) -> SelectionResult:
    """Pick the AutoML frameworks to run for a dataset, with no human input.

    Parameters
    ----------
    profile:
        Dataset profile (dict) as stored in ``profile_json``; may be None, in
        which case small-dataset defaults apply.
    budget_s:
        Explicit per-framework budget override; None selects by dataset size.
    availability:
        Availability records; defaults to a live environment scan.
    policy:
        Selection policy; defaults to ``configs/automl_selection.yaml``.
    wall_budget_s:
        Wall-clock cap override for the whole AutoML tier.

    Returns
    -------
    SelectionResult
        Ordered framework ids, the budget to run them with, and a reason for
        every selection and every skip.
    """
    policy = policy if policy is not None else load_policy()
    avail = availability if availability is not None else framework_availability()
    profile = profile or {}

    n_rows = int(profile.get("n_rows") or 0)
    size = str(profile.get("size_category") or "small")
    budget_by_size = policy.get("budget_by_size") or {}
    chosen_budget = int(
        budget_s or budget_by_size.get(size) or policy.get("default_budget_s") or 60
    )

    fw_policy: dict[str, Any] = policy.get("frameworks") or {}
    selected: dict[str, str] = {}
    skipped: dict[str, str] = {}
    candidates: list[tuple[int, str, dict]] = []

    for fw in avail:
        fid = fw["id"]
        limits = fw_policy.get(fid) or {}
        if not fw.get("os_supported", True):
            skipped[fid] = "unsupported on this OS"
            continue
        if not fw.get("installed", False):
            skipped[fid] = "environment not installed"
            continue
        max_rows = limits.get("max_rows")
        if n_rows and max_rows and n_rows > int(max_rows):
            skipped[fid] = f"dataset too large ({n_rows} rows > {int(max_rows)})"
            continue
        min_rows = limits.get("min_rows")
        if n_rows and min_rows and n_rows < int(min_rows):
            skipped[fid] = f"dataset too small ({n_rows} rows < {int(min_rows)})"
            continue
        candidates.append(
            (int(limits.get("priority", 99)), fw.get("kind") or "single_pipeline", fw)
        )

    candidates.sort(key=lambda item: item[0])

    # How many frameworks fit the wall-clock budget?
    wall = int(wall_budget_s or policy.get("total_wall_budget_s") or 900)
    multiplier = float(policy.get("wall_clock_multiplier") or 1.25)
    per_framework = max(1.0, chosen_budget * multiplier)
    capacity = min(
        int(policy.get("max_frameworks") or 4),
        max(1, int(wall // per_framework)),
    )

    if not candidates:
        if avail and not skipped:
            skipped["*"] = "no frameworks available"
        logger.info("auto-select: no eligible frameworks (%d skipped)", len(skipped))
        return SelectionResult([], chosen_budget, selected, skipped)

    # Pass 1: one framework per search-strategy kind (diversity), in priority order.
    picked: list[tuple[str, str]] = []
    seen_kinds: set[str] = set()
    for _prio, kind, fw in candidates:
        if len(picked) >= capacity:
            break
        if kind in seen_kinds:
            continue
        seen_kinds.add(kind)
        picked.append((fw["id"], f"auto-selected: representative of '{kind}' search strategy"))

    # Pass 2: fill remaining slots by priority.
    picked_ids = {fid for fid, _ in picked}
    for _prio, _kind, fw in candidates:
        if len(picked) >= capacity:
            break
        if fw["id"] in picked_ids:
            continue
        picked.append((fw["id"], "auto-selected: high priority within wall-clock budget"))
        picked_ids.add(fw["id"])

    for fid, reason in picked:
        selected[fid] = reason
    for _prio, _kind, fw in candidates:
        if fw["id"] not in picked_ids:
            skipped[fw["id"]] = "not needed: budget filled by higher-priority frameworks"

    frameworks = [fid for fid, _ in picked]
    logger.info(
        "auto-selected %d framework(s) at %ds budget: %s",
        len(frameworks),
        chosen_budget,
        ", ".join(frameworks),
    )
    return SelectionResult(frameworks, chosen_budget, selected, skipped)


def default_n_jobs(max_jobs: int = 4) -> int:
    """Sensible parallelism for AutoML workers without asking a human."""
    return max(1, min(max_jobs, os.cpu_count() or 1))
