from __future__ import annotations

from typing import Any, Dict

from pytest import MonkeyPatch

from app.core.config import settings
from app.services import llm


def test_llm_not_configured_without_key(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", None)
    assert llm.is_llm_configured() is False


def test_generate_grounded_answer_uses_httpx(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    monkeypatch.setattr(settings, "openai_model", "gpt-4o-mini")
    monkeypatch.setattr(settings, "openai_base_url", "https://api.openai.com/v1")

    captured: Dict[str, Any] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Dict[str, Any]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "Visitor parking is behind Building B. "
                                "Sources: [1]"
                            )
                        }
                    }
                ]
            }

    class FakeClient:
        def __init__(self, timeout: float) -> None:
            captured["timeout"] = timeout

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def post(self, url: str, headers: Dict[str, str], json: Dict[str, Any]) -> FakeResponse:
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr(llm.httpx, "Client", FakeClient)

    answer = llm.generate_grounded_answer(
        "Where is parking?",
        [{"source": "policy.txt", "text": "Visitor parking is behind Building B."}],
    )
    assert "Building B" in answer
    assert captured["url"].endswith("/chat/completions")
    assert "Bearer test-key" in captured["headers"]["Authorization"]
    user_content = captured["json"]["messages"][1]["content"]
    assert "Where is parking?" in user_content
    assert "[1] source=policy.txt" in user_content
