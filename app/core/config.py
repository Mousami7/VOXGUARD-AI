from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "VoxGuard-AI"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    audio_dir: Path = BASE_DIR / "data" / "audio"
    documents_dir: Path = BASE_DIR / "data" / "documents"
    chroma_dir: Path = BASE_DIR / "data" / "chroma"

    whisper_model: str = "base"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = Field(default=800, ge=50)
    chunk_overlap: int = Field(default=150, ge=0)
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self


settings = Settings()
