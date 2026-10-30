from __future__ import annotations

from fastapi.testclient import TestClient
from pytest import MonkeyPatch


def test_audio_transcribe_success(client: TestClient, monkeypatch: MonkeyPatch) -> None:
    def fake_transcribe(_path: str) -> dict:
        return {"language": "en", "transcription": "hello world"}

    monkeypatch.setattr(
        "app.api.audio.transcribe_audio",
        fake_transcribe,
    )
    response = client.post(
        "/api/v1/audio/transcribe",
        files={"file": ("sample.wav", b"fake-audio-bytes", "audio/wav")},
    )
    assert response.status_code == 200
    assert response.json() == {
        "filename": "sample.wav",
        "language": "en",
        "transcription": "hello world",
    }


def test_audio_rejects_missing_filename(client: TestClient) -> None:
    response = client.post(
        "/api/v1/audio/transcribe",
        files={"file": ("", b"data", "application/octet-stream")},
    )
    assert response.status_code in {400, 422}


def test_audio_rejects_unsupported_extension(client: TestClient) -> None:
    response = client.post(
        "/api/v1/audio/transcribe",
        files={"file": ("notes.txt", b"not audio", "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_audio_rejects_empty_file(client: TestClient) -> None:
    response = client.post(
        "/api/v1/audio/transcribe",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty."


def test_audio_rejects_oversized_file(client: TestClient) -> None:
    payload = b"x" * 2048
    response = client.post(
        "/api/v1/audio/transcribe",
        files={"file": ("large.wav", payload, "audio/wav")},
    )
    assert response.status_code == 413
    assert "maximum size" in response.json()["detail"]


def test_audio_transcription_error_returns_500(
    client: TestClient,
    monkeypatch: MonkeyPatch,
) -> None:
    def fail_transcribe(_path: str) -> dict:
        raise RuntimeError("whisper exploded")

    monkeypatch.setattr("app.api.audio.transcribe_audio", fail_transcribe)
    response = client.post(
        "/api/v1/audio/transcribe",
        files={"file": ("sample.wav", b"fake-audio-bytes", "audio/wav")},
    )
    assert response.status_code == 500
    assert response.json()["detail"] == "Transcription failed. Please try a different audio file."
