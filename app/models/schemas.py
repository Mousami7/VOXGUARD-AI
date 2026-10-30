from typing import List, Optional

from pydantic import BaseModel, Field


class TranscriptionResponse(BaseModel):
    filename: str
    language: Optional[str] = None
    transcription: str


class DocumentUploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    message: str


class DocumentListResponse(BaseModel):
    documents: List[str]


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=4, ge=1, le=10)
    transcription: Optional[str] = None


class SourceChunk(BaseModel):
    source: str
    text: str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceChunk]
