from __future__ import annotations

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from app.core.config import settings


def test_document_upload_success(client: TestClient, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("app.api.documents.extract_text", lambda _path: "policy text")
    monkeypatch.setattr("app.api.documents.index_document", lambda _name, _text: 2)

    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("policy.txt", b"policy text", "text/plain")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "policy.txt"
    assert body["chunks_indexed"] == 2
    assert body["message"] == "Document uploaded and indexed."
    assert (settings.documents_dir / "policy.txt").exists()


def test_document_rejects_unsupported_extension(client: TestClient) -> None:
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("notes.docx", b"data", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_document_rejects_empty_file(client: TestClient) -> None:
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty."


def test_document_rejects_oversized_file(client: TestClient) -> None:
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("large.txt", b"x" * 2048, "text/plain")},
    )
    assert response.status_code == 413


def test_document_rejects_overwrite(client: TestClient, monkeypatch: MonkeyPatch) -> None:
    existing = settings.documents_dir / "policy.txt"
    existing.write_text("already here", encoding="utf-8")
    monkeypatch.setattr("app.api.documents.extract_text", lambda _path: "new")
    monkeypatch.setattr("app.api.documents.index_document", lambda _name, _text: 1)

    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("policy.txt", b"new", "text/plain")},
    )
    assert response.status_code == 409
    assert existing.read_text(encoding="utf-8") == "already here"


def test_failed_extraction_does_not_keep_document(
    client: TestClient,
    monkeypatch: MonkeyPatch,
) -> None:
    def fail_extract(_path: str) -> str:
        raise ValueError("unreadable")

    monkeypatch.setattr("app.api.documents.extract_text", fail_extract)
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("broken.txt", b"cannot parse", "text/plain")},
    )
    assert response.status_code == 400
    assert not (settings.documents_dir / "broken.txt").exists()
    leftovers = list(settings.documents_dir.glob("*"))
    assert leftovers == []


def test_failed_indexing_does_not_keep_document(
    client: TestClient,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.api.documents.extract_text", lambda _path: "enough text")

    def fail_index(_name: str, _text: str) -> int:
        raise RuntimeError("chroma down")

    monkeypatch.setattr("app.api.documents.index_document", fail_index)
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("policy.txt", b"enough text", "text/plain")},
    )
    assert response.status_code == 500
    assert not (settings.documents_dir / "policy.txt").exists()
    leftovers = [path for path in settings.documents_dir.iterdir() if path.is_file()]
    assert leftovers == []


def test_empty_query_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/v1/documents/query",
        json={"question": "   "},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "A question is required."


def test_query_response_structure(client: TestClient, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.api.documents.query_knowledge",
        lambda question, top_k, transcription=None: {
            "question": question,
            "answer": "Based on the available sources:\n\nvisitor parking is behind Building B.",
            "sources": [
                {
                    "source": "policy.txt",
                    "text": "visitor parking is behind Building B.",
                    "score": 0.42,
                }
            ],
        },
    )
    response = client.post(
        "/api/v1/documents/query",
        json={"question": "Where is visitor parking?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "Where is visitor parking?"
    assert "Based on the available sources" in body["answer"]
    assert body["sources"][0]["source"] == "policy.txt"
    assert "text" in body["sources"][0]
    assert "score" in body["sources"][0]
