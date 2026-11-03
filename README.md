# VoxGuard-AI

Voice-aware AI application for processing audio recordings and documents.

This repository is an independent portfolio project. It currently provides a FastAPI backend that transcribes audio with Whisper, retrieves document passages with MiniLM + Chroma, and can generate a source-grounded LLM answer when an API key is configured.

## Current architecture

```
Create case
     |
     +-- upload audio ---- Whisper ---- store transcript ---- index (case-scoped)
     |
     +-- upload documents -- extract ---- index (case-scoped)
     |
     v
Ask a question on that case
     |
     v
Retrieve only that case's passages
     |
     v
If evidence is weak: refuse
If OPENAI_API_KEY is set: grounded LLM answer
Otherwise: extractive passages
```

Standalone `/api/v1/audio` and `/api/v1/documents` endpoints still work without a case.

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
│   ├── chroma/         # vector store (gitignored)
│   └── cases/          # case files and JSON store (gitignored)
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
| `POST` | `/api/v1/cases` | Create an investigation case |
| `GET` | `/api/v1/cases` | List cases (no transcript text) |
| `GET` | `/api/v1/cases/{id}` | Get one case, including transcript |
| `POST` | `/api/v1/cases/{id}/audio` | Transcribe audio and index it for the case |
| `POST` | `/api/v1/cases/{id}/documents` | Upload a document into the case |
| `POST` | `/api/v1/cases/{id}/query` | Ask a question using only that case's evidence |

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

## Cases (audio + documents together)

A case is one investigation, such as a claim file. Create a case, add one audio recording and one or more documents, then query that case. Retrieval is limited to that case's indexed transcript and documents.

```bash
CASE_ID=$(curl -s -X POST "http://127.0.0.1:8000/api/v1/cases" \
  -H "Content-Type: application/json" \
  -d '{"title":"Water leak claim"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")

curl -X POST "http://127.0.0.1:8000/api/v1/cases/$CASE_ID/audio" \
  -F "file=@data/audio/test.m4a"

curl -X POST "http://127.0.0.1:8000/api/v1/cases/$CASE_ID/documents" \
  -F "file=@data/documents/policy.txt"

curl -X POST "http://127.0.0.1:8000/api/v1/cases/$CASE_ID/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"Does the policy cover the damage described in the recording?"}'
```

A case accepts one transcript. Duplicate document names in the same case return HTTP 409. Case files are stored under `data/cases/` and are not committed.

You can still query global documents and optionally include a transcript so retrieval also searches speech text:

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

A LangGraph workflow that runs a case as explicit steps: transcribe → index → retrieve → grounded answer → citations. Whisper, MiniLM, Chroma, and the case API stay in place.
