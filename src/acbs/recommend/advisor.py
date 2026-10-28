"""Benchmark Advisor narration through the hosted Respan gateway.

One optional LLM step on top of the deterministic recommendation: the advisor
rewrites the stored output of :func:`acbs.recommend.explain.generate_explanation`
as plain prose. It never computes, re-ranks or invents a metric — every number
it may mention is passed in from the stored results.

Requests go to Respan's OpenAI-compatible Chat Completions endpoint
(``https://api.respan.ai/api/chat/completions``), so the project needs a single
server-side ``RESPAN_API_KEY`` and no provider SDK; the gateway records each call
as a trace. Without the key the advisor is unavailable and the app keeps serving
the template explanation.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from acbs.utils.logging import get_logger
from acbs.utils.paths import load_default_config

logger = get_logger(__name__)

DEFAULT_BASE_URL = "https://api.respan.ai/api"
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_MAX_TOKENS = 600
REQUEST_TIMEOUT_S = 30.0

_SYSTEM_PROMPT = (
    "You are the Benchmark Advisor of the Auto Classifier Benchmarking System. "
    "You explain one already-computed benchmark recommendation to a data scientist.\n"
    "Rules:\n"
    "- Use only the facts listed in the user message. Never invent, recompute, "
    "estimate or re-round a number that is not listed there.\n"
    "- Do not claim any benchmark was run, changed or re-run; the numbers are fixed.\n"
    "- Answer in 3 to 5 plain sentences, without headings or bullet lists.\n"
    "- If the question cannot be answered from the listed facts, say so and point "
    "back to the stored results."
)


def gateway_base_url() -> str:
    """Respan gateway base URL; ``RESPAN_BASE_URL`` overrides it for Enterprise."""
    return os.getenv("RESPAN_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def is_available() -> bool:
    """True when a server-side Respan key is configured."""
    return bool(os.getenv("RESPAN_API_KEY"))


def advisor_settings() -> dict[str, Any]:
    """Model and token budget from the ``advisor`` block of ``configs/default.yaml``."""
    try:
        cfg = load_default_config().get("advisor") or {}
    except Exception:  # a missing config must degrade to defaults, not break the advisor
        logger.warning("advisor config unavailable, using defaults")
        cfg = {}
    return {
        "model": str(cfg.get("model") or DEFAULT_MODEL),
        "max_tokens": int(cfg.get("max_tokens") or DEFAULT_MAX_TOKENS),
    }


def _facts(explanation: dict[str, Any]) -> str:
    """Flatten a stored explanation into the only facts the model may use."""
    facts = [
        f"headline: {explanation.get('headline')}",
        f"recommended model: {explanation.get('model_name')}",
        f"composite score: {explanation.get('score')}",
    ]
    stability = explanation.get("stability_pct")
    if stability is not None:
        facts.append(
            f"recommendation stability under +/-20% weight change: {float(stability):.0f}%"
        )
    for paragraph in explanation.get("paragraphs") or []:
        facts.append(f"finding: {paragraph}")
    for runner_up in explanation.get("why_not") or []:
        facts.append(f"runner-up: {runner_up}")
    for caveat in explanation.get("caveats") or []:
        facts.append(f"caveat: {caveat}")
    value = explanation.get("automl_value") or explanation.get("automl_value_analysis") or {}
    if isinstance(value, dict) and value.get("summary"):
        facts.append(f"automl value: {value['summary']}")
    return "\n".join(facts)


def build_advisor_messages(
    explanation: dict[str, Any], question: str | None = None
) -> list[dict[str, str]]:
    """Build the chat messages: system rules plus the grounded stored facts."""
    ask = question.strip() if question and question.strip() else "Explain this recommendation."
    user = f"Stored benchmark facts:\n{_facts(explanation)}\n\nUser question: {ask}"
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def narrate(
    explanation: dict[str, Any],
    *,
    question: str | None = None,
    timeout_s: float = REQUEST_TIMEOUT_S,
) -> str:
    """Send one grounded explanation request to the Respan gateway.

    The gateway logs the call, so this is the app's real traced LLM workflow.

    Args:
        explanation: Stored explanation produced by ``generate_explanation``.
        question: Optional user question; without it the advisor summarises.
        timeout_s: Request timeout in seconds.

    Returns:
        The advisor narrative, stripped of surrounding whitespace.

    Raises:
        RuntimeError: If ``RESPAN_API_KEY`` is unset or the gateway answers with
            an error status or an unexpected payload.
    """
    api_key = os.getenv("RESPAN_API_KEY")
    if not api_key:
        raise RuntimeError("RESPAN_API_KEY is not set")

    settings = advisor_settings()
    payload: dict[str, Any] = {
        "model": settings["model"],
        "messages": build_advisor_messages(explanation, question),
        "temperature": 0.2,
        "max_tokens": settings["max_tokens"],
        # Indexed on the gateway so this workflow is easy to filter in Logs.
        "metadata": {"app": "acbs", "workflow": "advisor-explanation"},
    }
    url = f"{gateway_base_url()}/chat/completions"
    response = httpx.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout_s,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Respan gateway returned {response.status_code}: {response.text[:300]}")
    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"unexpected Respan gateway response: {exc}") from exc
    return str(content).strip()
