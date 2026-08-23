from __future__ import annotations

import whisper

_whisper_model = None


def get_whisper_model():
    """Load the Whisper base model once and reuse it."""
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def transcribe_audio(audio_path: str) -> dict:
    """Transcribe a local audio file and return language plus text."""
    model = get_whisper_model()
    result = model.transcribe(audio_path, fp16=False)

    transcription = (result.get("text") or "").strip()
    language = result.get("language")

    return {
        "language": language,
        "transcription": transcription,
    }
