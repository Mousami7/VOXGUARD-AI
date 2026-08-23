# VoxGuard-AI

Voice-aware AI system for analyzing audio and related documents.

## Project structure

```
VoxGuard-AI/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── audio.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   └── transcription.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py
│   │
│   └── core/
│       ├── __init__.py
│       └── config.py
│
├── data/
│   ├── audio/
│   └── documents/
│
├── tests/
├── .venv/
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Run

```bash
uvicorn app.main:app --reload
```

- Health check: `GET /health`
- Transcribe audio: `POST /audio/transcribe`
