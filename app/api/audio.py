from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.models.schemas import TranscriptionResponse
from app.services.transcription import transcribe_audio

router = APIRouter(prefix="/api/v1/audio", tags=["audio"])

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_uploaded_audio(file: UploadFile = File(...)) -> TranscriptionResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="A filename is required.")

    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension or 'unknown'}. Allowed types: {allowed}",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        try:
            result = await run_in_threadpool(transcribe_audio, temp_path)
        except Exception:
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
