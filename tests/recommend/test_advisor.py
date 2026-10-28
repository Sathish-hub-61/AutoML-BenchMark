from __future__ import annotations

import pytest

from acbs.recommend import advisor

EXPLANATION = {
    "headline": "Recommended: Random Forest",
    "model_name": "random_forest",
    "score": 0.812,
    "stability_pct": 85.0,
    "paragraphs": ["Hold-out F1-macro is 0.951 and accuracy is 0.964."],
    "why_not": ["logistic_regression: lower composite score (0.700); F1=0.940."],
    "caveats": ["Inference latency is high (60.0 ms/1k rows) due to model ensembling."],
    "automl_value": {"summary": "AutoML gain is marginal over the classic baseline."},
}


class FakeResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self) -> dict:
        return self._payload


def _chat_payload(content: str = "  Random Forest wins on F1-macro.  ") -> dict:
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": content}}]}


def test_availability_follows_the_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("RESPAN_API_KEY", raising=False)
    assert advisor.is_available() is False
    monkeypatch.setenv("RESPAN_API_KEY", "test-key")
    assert advisor.is_available() is True


def test_base_url_defaults_to_respan_and_honours_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("RESPAN_BASE_URL", raising=False)
    assert advisor.gateway_base_url() == "https://api.respan.ai/api"
    monkeypatch.setenv("RESPAN_BASE_URL", "https://endpoint.respan.ai/api/")
    assert advisor.gateway_base_url() == "https://endpoint.respan.ai/api"


def test_messages_carry_only_the_stored_facts():
    messages = advisor.build_advisor_messages(EXPLANATION, "Why not logistic regression?")

    assert messages[0]["role"] == "system"
    assert "Never invent" in messages[0]["content"]
    user = messages[1]["content"]
    assert "0.951" in user
    assert "85%" in user
    assert "Why not logistic regression?" in user
    # the stored caveats and value analysis are available as grounding
    assert "Inference latency is high" in user
    assert "AutoML gain is marginal" in user


def test_narrate_posts_grounded_request_to_the_gateway(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RESPAN_API_KEY", "test-key")
    monkeypatch.delenv("RESPAN_BASE_URL", raising=False)
    captured: dict = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, json=json, timeout=timeout)
        return FakeResponse(payload=_chat_payload())

    monkeypatch.setattr(advisor.httpx, "post", fake_post)

    narrative = advisor.narrate(EXPLANATION, question="Why this model?")

    assert narrative == "Random Forest wins on F1-macro."
    assert captured["url"] == "https://api.respan.ai/api/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["metadata"]["workflow"] == "advisor-explanation"
    assert captured["json"]["model"] == advisor.advisor_settings()["model"]
    assert captured["json"]["messages"][1]["content"].startswith("Stored benchmark facts:")


def test_narrate_without_key_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("RESPAN_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="RESPAN_API_KEY"):
        advisor.narrate(EXPLANATION)


def test_narrate_surfaces_gateway_errors(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RESPAN_API_KEY", "test-key")
    monkeypatch.setattr(
        advisor.httpx,
        "post",
        lambda *a, **k: FakeResponse(status_code=429, text="rate limited"),
    )
    with pytest.raises(RuntimeError, match="429"):
        advisor.narrate(EXPLANATION)


def test_narrate_rejects_unexpected_payload(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RESPAN_API_KEY", "test-key")
    monkeypatch.setattr(advisor.httpx, "post", lambda *a, **k: FakeResponse(payload={"error": {}}))
    with pytest.raises(RuntimeError, match="unexpected"):
        advisor.narrate(EXPLANATION)
