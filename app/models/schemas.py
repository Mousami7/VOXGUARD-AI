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
    mode: str = "extractive"
    case_id: Optional[str] = None


class CaseCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class CaseSummary(BaseModel):
    id: str
    title: str
    created_at: str
    audio_filename: Optional[str] = None
    document_count: int
    has_transcript: bool


class CaseDetail(BaseModel):
    id: str
    title: str
    created_at: str
    audio_filename: Optional[str] = None
    transcript: Optional[str] = None
    transcript_language: Optional[str] = None
    documents: List[str]


class CaseListResponse(BaseModel):
    cases: List[CaseSummary]


class CaseAudioResponse(BaseModel):
    case_id: str
    filename: str
    language: Optional[str] = None
    transcription: str
    chunks_indexed: int


class CaseDocumentResponse(BaseModel):
    case_id: str
    filename: str
    chunks_indexed: int
    message: str


class CaseQueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=4, ge=1, le=10)
