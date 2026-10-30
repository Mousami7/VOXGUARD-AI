from __future__ import annotations

import pytest

from app.services.rag import chunk_text


def test_chunk_text_returns_empty_for_blank_input() -> None:
    assert chunk_text("   \n\t  ", chunk_size=20, overlap=5) == []


def test_chunk_text_keeps_short_text_as_one_chunk() -> None:
    assert chunk_text("hello world", chunk_size=50, overlap=5) == ["hello world"]


def test_chunk_text_uses_overlap() -> None:
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = chunk_text(text, chunk_size=10, overlap=4)
    assert chunks[0] == "abcdefghij"
    assert chunks[1].startswith("ghij")
    assert chunks[1][:4] == chunks[0][-4:]


def test_chunk_text_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_text("abcdef", chunk_size=4, overlap=4)
