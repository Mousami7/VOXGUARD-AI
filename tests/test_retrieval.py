from __future__ import annotations

from typing import Any, Dict, List

from pytest import MonkeyPatch

from app.core.config import settings
from app.services import rag
from app.services.llm import REFUSAL_MESSAGE


def test_query_knowledge_response_structure(monkeypatch: MonkeyPatch) -> None:
    def fake_documents(_question: str, _top_k: int) -> List[Dict[str, Any]]:
        return [
            {
                "source": "policy.txt",
                "text": "Visitor parking is behind Building B.",
                "score": 0.51,
            }
        ]

    monkeypatch.setattr(rag, "_search_documents", fake_documents)
    monkeypatch.setattr(settings, "openai_api_key", None)
    result = rag.query_knowledge("Where is parking?", top_k=4)

    assert set(result.keys()) == {"question", "answer", "sources", "mode"}
    assert result["question"] == "Where is parking?"
    assert result["answer"].startswith("Based on the available sources:")
    assert result["mode"] == "extractive"
    assert result["sources"][0]["source"] == "policy.txt"
    assert "Building B" in result["sources"][0]["text"]
    assert isinstance(result["sources"][0]["score"], float)


def test_query_knowledge_empty_sources(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(rag, "_search_documents", lambda _q, _k: [])
    result = rag.query_knowledge("unknown topic")
    assert result["sources"] == []
    assert result["mode"] == "refused"
    assert result["answer"] == rag.NO_SOURCES_MESSAGE


def test_query_knowledge_weak_evidence_is_refused(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        rag,
        "_search_documents",
        lambda _q, _k: [{"source": "policy.txt", "text": "unrelated", "score": 0.05}],
    )
    monkeypatch.setattr(settings, "min_evidence_score", 0.25)
    result = rag.query_knowledge("Where is parking?")
    assert result["mode"] == "refused"
    assert result["answer"] == REFUSAL_MESSAGE
    assert result["sources"][0]["source"] == "policy.txt"


def test_query_knowledge_merges_transcript_sources(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        rag,
        "_search_documents",
        lambda _q, _k: [
            {"source": "policy.txt", "text": "from document", "score": 0.2}
        ],
    )
    monkeypatch.setattr(
        rag,
        "_search_transcription",
        lambda _q, _t, _k: [
            {"source": "transcription", "text": "from speech", "score": 0.9}
        ],
    )
    monkeypatch.setattr(settings, "openai_api_key", None)
    result = rag.query_knowledge("question", transcription="ignored by mock")
    assert result["sources"][0]["source"] == "transcription"
    assert "from speech" in result["answer"]
    assert result["mode"] == "extractive"


def test_query_knowledge_uses_grounded_llm(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        rag,
        "_search_documents",
        lambda _q, _k: [
            {
                "source": "policy.txt",
                "text": "Visitor parking is behind Building B.",
                "score": 0.51,
            }
        ],
    )
    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    monkeypatch.setattr(
        rag,
        "generate_grounded_answer",
        lambda _question, _sources: "Visitor parking is behind Building B. Sources: [1]",
    )
    result = rag.query_knowledge("Where is parking?")
    assert result["mode"] == "generative"
    assert result["answer"].startswith("Visitor parking is behind Building B.")
    assert result["sources"][0]["source"] == "policy.txt"


def test_query_knowledge_falls_back_when_llm_fails(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        rag,
        "_search_documents",
        lambda _q, _k: [
            {"source": "policy.txt", "text": "Visitor parking is behind Building B.", "score": 0.51}
        ],
    )
    monkeypatch.setattr(settings, "openai_api_key", "test-key")

    def fail_llm(_question: str, _sources: List[Dict[str, Any]]) -> str:
        raise RuntimeError("provider down")

    monkeypatch.setattr(rag, "generate_grounded_answer", fail_llm)
    result = rag.query_knowledge("Where is parking?")
    assert result["mode"] == "extractive"
    assert result["answer"].startswith("Based on the available sources:")
