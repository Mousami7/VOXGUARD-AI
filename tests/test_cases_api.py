from __future__ import annotations

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from app.core.config import settings


def _create_case(client: TestClient, title: str = "Claim 1001") -> str:
    response = client.post("/api/v1/cases", json={"title": title})
    assert response.status_code == 200
    return response.json()["id"]


def test_create_and_list_cases(client: TestClient) -> None:
    case_id = _create_case(client, "Water leak claim")
    listed = client.get("/api/v1/cases")
    assert listed.status_code == 200
    body = listed.json()
    assert body["cases"][0]["id"] == case_id
    assert body["cases"][0]["title"] == "Water leak claim"
    assert body["cases"][0]["has_transcript"] is False
    assert "transcript" not in body["cases"][0]

    detail = client.get("/api/v1/cases/{case_id}".format(case_id=case_id))
    assert detail.status_code == 200
    assert detail.json()["documents"] == []
    assert detail.json()["transcript"] is None


def test_case_not_found(client: TestClient) -> None:
    response = client.get("/api/v1/cases/not-a-real-case")
    assert response.status_code == 404


def test_case_audio_transcribes_and_indexes(
    client: TestClient,
    monkeypatch: MonkeyPatch,
) -> None:
    case_id = _create_case(client)
    monkeypatch.setattr(
        "app.api.cases.transcribe_audio",
        lambda _path: {"language": "en", "transcription": "The basement flooded last night."},
    )
    monkeypatch.setattr("app.api.cases.index_document", lambda *_args: 1)

    response = client.post(
        "/api/v1/cases/{case_id}/audio".format(case_id=case_id),
        files={"file": ("claim.wav", b"fake-audio", "audio/wav")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["case_id"] == case_id
    assert body["transcription"] == "The basement flooded last night."
    assert body["chunks_indexed"] == 1

    detail = client.get("/api/v1/cases/{case_id}".format(case_id=case_id)).json()
    assert detail["audio_filename"] == "claim.wav"
    assert detail["transcript"] == "The basement flooded last night."


def test_case_rejects_second_audio(client: TestClient, monkeypatch: MonkeyPatch) -> None:
    case_id = _create_case(client)
    monkeypatch.setattr(
        "app.api.cases.transcribe_audio",
        lambda _path: {"language": "en", "transcription": "first recording"},
    )
    monkeypatch.setattr("app.api.cases.index_document", lambda *_args: 1)
    first = client.post(
        "/api/v1/cases/{case_id}/audio".format(case_id=case_id),
        files={"file": ("one.wav", b"audio-one", "audio/wav")},
    )
    assert first.status_code == 200
    second = client.post(
        "/api/v1/cases/{case_id}/audio".format(case_id=case_id),
        files={"file": ("two.wav", b"audio-two", "audio/wav")},
    )
    assert second.status_code == 409


def test_case_document_upload(client: TestClient, monkeypatch: MonkeyPatch) -> None:
    case_id = _create_case(client)
    monkeypatch.setattr("app.api.cases.extract_text", lambda _path: "Policy covers water damage.")
    monkeypatch.setattr("app.api.cases.index_document", lambda *_args: 2)

    response = client.post(
        "/api/v1/cases/{case_id}/documents".format(case_id=case_id),
        files={"file": ("policy.txt", b"Policy covers water damage.", "text/plain")},
    )
    assert response.status_code == 200
    assert response.json()["filename"] == "policy.txt"
    assert (settings.cases_dir / case_id / "policy.txt").exists()

    duplicate = client.post(
        "/api/v1/cases/{case_id}/documents".format(case_id=case_id),
        files={"file": ("policy.txt", b"Policy covers water damage.", "text/plain")},
    )
    assert duplicate.status_code == 409


def test_failed_case_document_does_not_remain(
    client: TestClient,
    monkeypatch: MonkeyPatch,
) -> None:
    case_id = _create_case(client)
    monkeypatch.setattr(
        "app.api.cases.extract_text",
        lambda _path: (_ for _ in ()).throw(ValueError("bad pdf")),
    )
    response = client.post(
        "/api/v1/cases/{case_id}/documents".format(case_id=case_id),
        files={"file": ("broken.pdf", b"%PDF-fake", "application/pdf")},
    )
    assert response.status_code == 400
    leftovers = list((settings.cases_dir / case_id).glob("*")) if (settings.cases_dir / case_id).exists() else []
    assert leftovers == []


def test_case_query_uses_case_id(client: TestClient, monkeypatch: MonkeyPatch) -> None:
    case_id = _create_case(client)
    captured = {}

    def fake_query(question, top_k, transcription=None, case_id=None):
        captured["question"] = question
        captured["case_id"] = case_id
        captured["transcription"] = transcription
        return {
            "question": question,
            "answer": "Policy covers water damage. Sources: [1]",
            "sources": [{"source": "policy.txt", "text": "Policy covers water damage.", "score": 0.6}],
            "mode": "generative",
        }

    monkeypatch.setattr("app.api.cases.query_knowledge", fake_query)
    response = client.post(
        "/api/v1/cases/{case_id}/query".format(case_id=case_id),
        json={"question": "Is water damage covered?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["case_id"] == case_id
    assert captured["case_id"] == case_id
    assert captured["transcription"] is None
    assert captured["question"] == "Is water damage covered?"


def test_case_query_rejects_empty_question(client: TestClient) -> None:
    case_id = _create_case(client)
    response = client.post(
        "/api/v1/cases/{case_id}/query".format(case_id=case_id),
        json={"question": "   "},
    )
    assert response.status_code == 400
