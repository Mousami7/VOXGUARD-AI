from fastapi import FastAPI

from app.api.audio import router as audio_router

app = FastAPI(
    title="VoxGuard AI",
    description="AI-powered speech intelligence and RAG platform",
    version="0.1.0",
)

app.include_router(audio_router)


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
