from fastapi import FastAPI

from app.api.audio import router as audio_router
from app.api.documents import router as documents_router
from app.core.config import settings
from app.core.logging import setup_logging
from app.core.middleware import RequestContextMiddleware

setup_logging()

app = FastAPI(
    title="VoxGuard AI",
    description="AI-powered speech intelligence and RAG platform",
    version=settings.app_version,
)

app.add_middleware(RequestContextMiddleware)
app.include_router(audio_router)
app.include_router(documents_router)


@app.get("/")
def root():
    return {
        "name": "VoxGuard AI",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }
