from __future__ import annotations

from typing import Any, Dict, List

from pytest import MonkeyPatch

from app.services import rag


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
    result = rag.query_knowledge("Where is parking?", top_k=4)

    assert set(result.keys()) == {"question", "answer", "sources"}
    assert result["question"] == "Where is parking?"
    assert result["answer"].startswith("Based on the available sources:")
    assert result["sources"][0]["source"] == "policy.txt"
    assert "Building B" in result["sources"][0]["text"]
    assert isinstance(result["sources"][0]["score"], float)


def test_query_knowledge_empty_sources(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(rag, "_search_documents", lambda _q, _k: [])
    result = rag.query_knowledge("unknown topic")
    assert result["sources"] == []
    assert result["answer"] == (
        "I could not find relevant information in the uploaded documents."
    )


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
    result = rag.query_knowledge("question", transcription="ignored by mock")
    assert result["sources"][0]["source"] == "transcription"
    assert "from speech" in result["answer"]
