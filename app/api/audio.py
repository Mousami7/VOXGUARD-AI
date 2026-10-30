from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.core.logging import get_logger
from app.core.uploads import read_upload_bounded, validate_upload_filename
from app.models.schemas import TranscriptionResponse
from app.services.transcription import transcribe_audio

router = APIRouter(prefix="/api/v1/audio", tags=["audio"])
logger = get_logger("api.audio")

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_uploaded_audio(file: UploadFile = File(...)) -> TranscriptionResponse:
    _, extension = validate_upload_filename(file.filename, ALLOWED_EXTENSIONS)
    contents = await read_upload_bounded(file)

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        try:
            result = await run_in_threadpool(transcribe_audio, temp_path)
        except Exception:
            logger.exception(
                "audio_transcribe_failed error_category=transcription filename=%s",
                Path(file.filename).name,
            )
            raise HTTPException(
                status_code=500,
                detail="Transcription failed. Please try a different audio file.",
            )

        return TranscriptionResponse(
            filename=file.filename,
            language=result.get("language"),
            transcription=result.get("transcription", ""),
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
