# VoxGuard-AI

Voice-aware AI application for processing audio recordings and documents.

This repository is an independent portfolio project. It currently provides a FastAPI backend that transcribes audio with Whisper and retrieves relevant document (and optional transcript) passages with extractive RAG.

## Current architecture

```
Upload audio                    Upload document
     |                                |
     v                                v
Whisper (`base`)                 Extract TXT/MD/PDF
     |                                |
     v                                v
Transcript JSON                  Chunk + MiniLM embeddings
                                      |
                                      v
                                 Persistent ChromaDB
                                      |
Ask a question -----------------------+
                                      |
                                      v
                         Retrieve top-k passages
                         (extractive answer, not an LLM)
```

```
VoxGuard-AI/
├── app/
│   ├── main.py
│   ├── api/            # HTTP routes
│   ├── services/       # Whisper, document extraction, RAG
│   ├── models/         # Pydantic schemas
│   └── core/           # Settings, logging, upload helpers
├── data/
│   ├── audio/          # local audio (gitignored)
│   ├── documents/      # uploaded documents (gitignored)
│   └── chroma/         # vector store (gitignored)
├── tests/
├── .env.example
├── pytest.ini
├── README.md
└── requirements.txt
```

## Python version and installation

This project currently runs on **Python 3.9.6** with the existing `.venv`.

```bash
cd "/Users/mousamisoni/Documents/AI Portfolio Project/VoxGuard-AI"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

FFmpeg is required for Whisper to decode audio (`brew install ffmpeg` on macOS).

## Run FastAPI

```bash
source .venv/bin/activate
uvicorn app.main:app --reload
```

The API is served at [http://127.0.0.1:8000](http://127.0.0.1:8000). Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Liveness: `{"name": "VoxGuard AI", "status": "running"}` |
| `GET` | `/health` | Health: `{"status": "healthy"}` |
| `POST` | `/api/v1/audio/transcribe` | Upload audio and return Whisper transcription |
| `POST` | `/api/v1/documents/upload` | Upload `.txt`, `.md`, or `.pdf` and index it |
| `GET` | `/api/v1/documents` | List indexed document filenames |
| `POST` | `/api/v1/documents/query` | Retrieve extractive passages for a question |

Audio types: `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`.

Default upload limit: 10 MiB (`MAX_UPLOAD_BYTES`). Duplicate document filenames return HTTP 409.

Example transcription:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/audio/transcribe" \
  -F "file=@data/audio/test.m4a"
```

Example document query:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/documents/upload" \
  -F "file=@data/documents/policy.txt"

curl -X POST "http://127.0.0.1:8000/api/v1/documents/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"Where is visitor parking?"}'
```

You can optionally include a transcript so retrieval also searches speech text:

```json
{
  "question": "What is the emergency contact?",
  "transcription": "The speaker mentioned the front desk."
}
```

## How document indexing works

1. The file type and size are validated.
2. Text is extracted (UTF-8 for TXT/MD, pypdf for PDF).
3. Text is split into overlapping character chunks (`CHUNK_SIZE`, `CHUNK_OVERLAP`).
4. Each chunk is embedded with sentence-transformers MiniLM (`EMBEDDING_MODEL`).
5. Embeddings are stored in a persistent Chroma collection under `data/chroma`.

The first embedding-model load may download MiniLM weights. Whisper `base` is loaded once and reused.

## Why RAG is extractive, not generative

`POST /api/v1/documents/query` returns the most similar stored passages joined together. It does **not** call an LLM to rewrite an answer. That keeps responses source-grounded and avoids hallucination until Phase 2.

Document scores come from Chroma L2 distance (`1 / (1 + distance)`). Optional transcript scores are cosine similarity. They are sorted together for the current API, but they are **not** on the same scale.

## Tests

```bash
source .venv/bin/activate
pytest
```

Tests mock Whisper, MiniLM, and Chroma at the service boundary. They use temporary directories and do not write into your real `data/` folders.

## Current limitations

- No LLM generation, LangChain, or LangGraph.
- No authentication, case IDs, or evidence audit trail.
- No frontend, Docker, or CI/CD.
- Health checks are liveness only; they do not verify model files.
- Python 3.9.6; `requirements.txt` is a full environment freeze.

## Planned Phase 2

Add source-grounded LLM answers on top of the current retriever: generate a short answer only from retrieved passages, include citations, and refuse when evidence is weak. Whisper, MiniLM, and Chroma stay in place.
