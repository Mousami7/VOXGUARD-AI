# VoxGuard-AI

Voice-aware AI application for processing audio recordings and documents.

This repository is an independent portfolio project. It currently provides a FastAPI backend that transcribes audio with Whisper, retrieves document passages with MiniLM + Chroma, and can generate a source-grounded LLM answer when an API key is configured.

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
                                      |
                                      v
                    If evidence is weak: refuse
                    If OPENAI_API_KEY is set: grounded LLM answer
                    Otherwise: extractive passages
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
| `POST` | `/api/v1/documents/query` | Retrieve passages and return a grounded answer |

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

## How grounded answers work

`POST /api/v1/documents/query` still retrieves the same passages. Then:

1. If no passages are found, it refuses.
2. If the best retrieval score is below `MIN_EVIDENCE_SCORE`, it refuses instead of guessing.
3. If `OPENAI_API_KEY` is set, an LLM writes a short answer using **only** those passages and cites them as `Sources: [1], [2]`.
4. If no API key is set, or the LLM call fails, it falls back to the previous extractive answer (`Based on the available sources:` plus the passages).

The JSON now includes `mode`: `generative`, `extractive`, or `refused`. `question`, `answer`, and `sources` are unchanged.

Add this to `.env` to enable generation (do not commit the key):

```bash
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4o-mini
```

Document scores come from Chroma L2 distance (`1 / (1 + distance)`). Optional transcript scores are cosine similarity. They are sorted together, but they are **not** on the same scale.

## Tests

```bash
source .venv/bin/activate
pytest
```

Tests mock Whisper, MiniLM, Chroma, and the LLM HTTP call. They use temporary directories and do not write into your real `data/` folders.

## Current limitations

- LLM answers need `OPENAI_API_KEY`. Without it, query stays extractive.
- No LangChain or LangGraph yet.
- No authentication, case IDs, or evidence audit trail.
- No frontend, Docker, or CI/CD.
- Health checks are liveness only; they do not verify model files.
- Python 3.9.6; `requirements.txt` is a full environment freeze.

## Planned next

A small LangGraph workflow that ties one case together: transcribe → index → retrieve → grounded answer → citations. Whisper, MiniLM, and Chroma stay in place.
