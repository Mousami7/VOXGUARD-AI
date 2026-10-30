from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings


@pytest.fixture
def tmp_data_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    audio_dir = tmp_path / "audio"
    documents_dir = tmp_path / "documents"
    chroma_dir = tmp_path / "chroma"
    audio_dir.mkdir()
    documents_dir.mkdir()
    chroma_dir.mkdir()
    monkeypatch.setattr(settings, "audio_dir", audio_dir)
    monkeypatch.setattr(settings, "documents_dir", documents_dir)
    monkeypatch.setattr(settings, "chroma_dir", chroma_dir)
    monkeypatch.setattr(settings, "max_upload_bytes", 1024)
    return tmp_path


@pytest.fixture
def client(tmp_data_dirs: Path) -> TestClient:
    from app.main import app

    return TestClient(app)
