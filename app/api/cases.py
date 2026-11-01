from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.logging import get_logger
from app.core.uploads import read_upload_bounded, validate_upload_filename
from app.models.schemas import (
    CaseAudioResponse,
    CaseCreateRequest,
    CaseDetail,
    CaseDocumentResponse,
    CaseListResponse,
    CaseQueryRequest,
    CaseSummary,
    QueryResponse,
)
from app.services.cases import create_case, get_case, list_cases, summarize_case, update_case
from app.services.documents import extract_text
from app.services.rag import index_document, query_knowledge
from app.services.transcription import transcribe_audio

router = APIRouter(prefix="/api/v1/cases", tags=["cases"])
logger = get_logger("api.cases")

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
DOCUMENT_EXTENSIONS = {".txt", ".md", ".pdf"}
TRANSCRIPT_SOURCE = "transcription"


def _require_case(case_id: str) -> dict:
    try:
        return get_case(case_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Case not found.")


@router.post("", response_model=CaseDetail)
def create_investigation_case(payload: CaseCreateRequest) -> CaseDetail:
    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="A case title is required.")
    return CaseDetail(**create_case(title))


@router.get("", response_model=CaseListResponse)
def list_investigation_cases() -> CaseListResponse:
    return CaseListResponse(cases=[CaseSummary(**summarize_case(item)) for item in list_cases()])


@router.get("/{case_id}", response_model=CaseDetail)
def get_investigation_case(case_id: str) -> CaseDetail:
    return CaseDetail(**_require_case(case_id))


@router.post("/{case_id}/audio", response_model=CaseAudioResponse)
async def add_case_audio(case_id: str, file: UploadFile = File(...)) -> CaseAudioResponse:
    case = _require_case(case_id)
    if case.get("transcript"):
        raise HTTPException(
            status_code=409,
            detail="This case already has a transcript. Create a new case to add different audio.",
        )

    filename, extension = validate_upload_filename(file.filename, AUDIO_EXTENSIONS)
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
                "case_audio_failed error_category=transcription case_id=%s filename=%s",
                case_id,
                filename,
            )
            raise HTTPException(
                status_code=500,
                detail="Transcription failed. Please try a different audio file.",
            )

        transcription = (result.get("transcription") or "").strip()
        if not transcription:
            raise HTTPException(
                status_code=400,
                detail="No speech could be transcribed from this audio file.",
            )

        try:
            chunk_count = await run_in_threadpool(
                index_document,
                TRANSCRIPT_SOURCE,
                transcription,
                case_id,
                TRANSCRIPT_SOURCE,
            )
        except Exception:
            logger.exception(
                "case_transcript_index_failed error_category=indexing case_id=%s",
                case_id,
            )
            raise HTTPException(
                status_code=500,
                detail="Transcript indexing failed. Please try again.",
            )

        update_case(
            case_id,
            audio_filename=filename,
            transcript=transcription,
            transcript_language=result.get("language"),
        )
        return CaseAudioResponse(
            case_id=case_id,
            filename=filename,
            language=result.get("language"),
            transcription=transcription,
            chunks_indexed=chunk_count,
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@router.post("/{case_id}/documents", response_model=CaseDocumentResponse)
async def add_case_document(case_id: str, file: UploadFile = File(...)) -> CaseDocumentResponse:
    case = _require_case(case_id)
    filename, extension = validate_upload_filename(file.filename, DOCUMENT_EXTENSIONS)
    if filename in (case.get("documents") or []):
        raise HTTPException(
            status_code=409,
            detail="A document with this filename already exists in the case.",
        )

    contents = await read_upload_bounded(file)
    case_dir = settings.cases_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
            dir=str(case_dir),
        ) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        try:
            text = extract_text(temp_path)
        except Exception:
            logger.exception(
                "case_document_extract_failed error_category=indexing case_id=%s filename=%s",
                case_id,
                filename,
            )
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from this document.",
            )
        if not text:
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from this document.",
            )

        try:
            chunk_count = await run_in_threadpool(
                index_document,
                filename,
                text,
                case_id,
                filename,
            )
        except Exception:
            logger.exception(
                "case_document_index_failed error_category=indexing case_id=%s filename=%s",
                case_id,
                filename,
            )
            raise HTTPException(
                status_code=500,
                detail="Document indexing failed. Please try a different file.",
            )

        destination = case_dir / filename
        Path(temp_path).replace(destination)
        temp_path = None
        documents = list(case.get("documents") or [])
        documents.append(filename)
        update_case(case_id, documents=documents)
        return CaseDocumentResponse(
            case_id=case_id,
            filename=filename,
            chunks_indexed=chunk_count,
            message="Document uploaded and indexed for this case.",
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@router.post("/{case_id}/query", response_model=QueryResponse)
async def query_case(case_id: str, payload: CaseQueryRequest) -> QueryResponse:
    _require_case(case_id)
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="A question is required.")

    try:
        result = await run_in_threadpool(
            query_knowledge,
            question,
            payload.top_k,
            None,
            case_id,
        )
    except Exception:
        logger.exception("case_query_failed error_category=retrieval case_id=%s", case_id)
        raise HTTPException(
            status_code=500,
            detail="Document search failed. Please try again.",
        )

    result["case_id"] = case_id
    return QueryResponse(**result)
