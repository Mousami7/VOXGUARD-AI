from __future__ import annotations

import time
from typing import Any, Dict, Optional

import whisper

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("transcription")

_whisper_model = None


def get_whisper_model():
    """Load the configured Whisper model once and reuse it."""
    global _whisper_model
    if _whisper_model is None:
        logger.info("whisper_model_loading model=%s", settings.whisper_model)
        _whisper_model = whisper.load_model(settings.whisper_model)
        logger.info("whisper_model_loaded model=%s", settings.whisper_model)
    return _whisper_model


def transcribe_audio(audio_path: str) -> Dict[str, Optional[str]]:
    """Transcribe a local audio file and return language plus text."""
    started = time.perf_counter()
    try:
        model = get_whisper_model()
        result: Dict[str, Any] = model.transcribe(audio_path, fp16=False)
        transcription = (result.get("text") or "").strip()
        language = result.get("language")
        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.info(
            "transcription_completed error_category=none duration_ms=%s",
            duration_ms,
        )
        return {
            "language": language,
            "transcription": transcription,
        }
    except Exception:
        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.exception(
            "transcription_failed error_category=transcription duration_ms=%s",
            duration_ms,
        )
        raise
