from typing import Optional

from pydantic import BaseModel


class TranscriptionResponse(BaseModel):
    filename: str
    language: Optional[str] = None
    transcription: str
