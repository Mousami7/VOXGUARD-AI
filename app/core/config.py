from pathlib import Path

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

    audio_dir: Path = BASE_DIR / "data" / "audio"
    documents_dir: Path = BASE_DIR / "data" / "documents"


settings = Settings()
