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
    DocumentListResponse,
    DocumentUploadResponse,
    QueryRequest,
    QueryResponse,
)
from app.services.documents import extract_text
from app.services.rag import index_document, list_indexed_documents, query_knowledge

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])
logger = get_logger("api.documents")

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    filename, extension = validate_upload_filename(file.filename, ALLOWED_EXTENSIONS)
    contents = await read_upload_bounded(file)

    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    destination = settings.documents_dir / filename
    if destination.exists():
        raise HTTPException(
            status_code=409,
            detail="A document with this filename already exists.",
        )

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
            dir=str(settings.documents_dir),
        ) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        try:
            text = extract_text(temp_path)
        except HTTPException:
            raise
        except Exception:
            logger.exception(
                "document_extract_failed error_category=indexing filename=%s",
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
            chunk_count = await run_in_threadpool(index_document, filename, text)
        except HTTPException:
            raise
        except Exception:
            logger.exception(
                "document_index_failed error_category=indexing filename=%s",
                filename,
            )
            raise HTTPException(
                status_code=500,
                detail="Document indexing failed. Please try a different file.",
            )

        Path(temp_path).replace(destination)
        temp_path = None
        return DocumentUploadResponse(
            filename=filename,
            chunks_indexed=chunk_count,
            message="Document uploaded and indexed.",
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@router.get("", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    return DocumentListResponse(documents=list_indexed_documents())


@router.post("/query", response_model=QueryResponse)
async def query_documents(payload: QueryRequest) -> QueryResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="A question is required.")

    try:
        result = await run_in_threadpool(
            query_knowledge,
            question,
            payload.top_k,
            payload.transcription,
        )
    except Exception:
        logger.exception("document_query_failed error_category=retrieval")
        raise HTTPException(
            status_code=500,
            detail="Document search failed. Please try again.",
        )

    return QueryResponse(**result)
