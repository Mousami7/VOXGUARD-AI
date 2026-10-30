from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("rag")

_embedding_model = None
_chroma_client = None
_collection = None

COLLECTION_NAME = "voxguard_documents"


def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("embedding_model_loading model=%s", settings.embedding_model)
        _embedding_model = SentenceTransformer(settings.embedding_model)
        logger.info("embedding_model_loaded model=%s", settings.embedding_model)
    return _embedding_model


def get_collection():
    """Open the local Chroma collection once and reuse it."""
    global _chroma_client, _collection
    if _collection is None:
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        _collection = _chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


def chunk_text(
    text: str,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> List[str]:
    size = settings.chunk_size if chunk_size is None else chunk_size
    step_overlap = settings.chunk_overlap if overlap is None else overlap
    if size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if step_overlap >= size:
        raise ValueError("chunk overlap must be smaller than chunk size")

    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(cleaned):
        end = start + size
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start = end - step_overlap
    return chunks


def _cosine_similarity(query_vector: Any, chunk_vectors: Any) -> List[float]:
    query = np.array(query_vector)
    chunks = np.array(chunk_vectors)
    query_norm = np.linalg.norm(query)
    chunk_norms = np.linalg.norm(chunks, axis=1)
    denom = chunk_norms * query_norm
    scores = np.zeros(len(chunks))
    valid = denom > 0
    scores[valid] = (chunks[valid] @ query) / denom[valid]
    return scores.tolist()


def _delete_document_chunks(filename: str) -> None:
    collection = get_collection()
    existing = collection.get(where={"filename": filename})
    ids = existing.get("ids") if existing else None
    if ids:
        collection.delete(ids=ids)


def index_document(filename: str, text: str) -> int:
    """Split a document into chunks, embed them, and store them."""
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No text could be extracted from this document.")

    started = time.perf_counter()
    model = get_embedding_model()
    collection = get_collection()
    embeddings = model.encode(chunks).tolist()
    ids = [f"{filename}::{index}" for index in range(len(chunks))]
    metadatas = [{"filename": filename, "source": filename} for _ in chunks]

    _delete_document_chunks(filename)
    try:
        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
    except Exception:
        _delete_document_chunks(filename)
        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.exception(
            "document_index_failed error_category=indexing filename=%s duration_ms=%s",
            filename,
            duration_ms,
        )
        raise

    duration_ms = round((time.perf_counter() - started) * 1000, 1)
    logger.info(
        "document_indexed error_category=none filename=%s chunk_count=%s duration_ms=%s",
        filename,
        len(chunks),
        duration_ms,
    )
    return len(chunks)


def list_indexed_documents() -> List[str]:
    collection = get_collection()
    stored = collection.get()
    filenames: List[str] = []
    for metadata in stored.get("metadatas") or []:
        name = metadata.get("filename")
        if name and name not in filenames:
            filenames.append(name)
    return filenames


def _search_documents(question: str, top_k: int) -> List[Dict[str, Any]]:
    collection = get_collection()
    if collection.count() == 0:
        return []

    model = get_embedding_model()
    query_embedding = model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    matches: List[Dict[str, Any]] = []
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    for text, metadata, distance in zip(documents, metadatas, distances):
        # Chroma default space is L2. Convert distance to a 0-1 style score.
        # This is NOT comparable to cosine scores from transcript search.
        score = 1 / (1 + float(distance))
        matches.append(
            {
                "source": metadata.get("source") or metadata.get("filename") or "document",
                "text": text,
                "score": round(score, 4),
            }
        )
    return matches


def _search_transcription(question: str, transcription: str, top_k: int) -> List[Dict[str, Any]]:
    chunks = chunk_text(transcription)
    if not chunks:
        return []

    model = get_embedding_model()
    query_embedding = model.encode([question])[0]
    chunk_embeddings = model.encode(chunks)
    scores = _cosine_similarity(query_embedding, chunk_embeddings)

    ranked = sorted(zip(chunks, scores), key=lambda item: item[1], reverse=True)
    matches: List[Dict[str, Any]] = []
    for text, score in ranked[:top_k]:
        # Cosine similarity in [-1, 1]. Do not mix this with Chroma L2 scores
        # as if they used the same scale.
        matches.append(
            {
                "source": "transcription",
                "text": text,
                "score": round(float(score), 4),
            }
        )
    return matches


def query_knowledge(
    question: str,
    top_k: int = 4,
    transcription: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve the most relevant document (and optional transcript) chunks.

    Document hits use Chroma L2 distance converted with 1/(1+d).
    Transcript hits use cosine similarity. Those scores are merged by numeric
    sort for compatibility, but they are not calibrated to the same scale.
    """
    started = time.perf_counter()
    try:
        matches = _search_documents(question, top_k)

        if transcription and transcription.strip():
            matches.extend(_search_transcription(question, transcription, top_k))

        matches.sort(key=lambda item: item["score"], reverse=True)
        matches = matches[:top_k]

        if not matches:
            answer = "I could not find relevant information in the uploaded documents."
        else:
            excerpts = [match["text"] for match in matches]
            answer = "Based on the available sources:\n\n" + "\n\n".join(excerpts)

        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.info(
            "query_completed error_category=none source_count=%s duration_ms=%s",
            len(matches),
            duration_ms,
        )
        return {
            "question": question,
            "answer": answer,
            "sources": matches,
        }
    except Exception:
        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.exception(
            "query_failed error_category=retrieval duration_ms=%s",
            duration_ms,
        )
        raise
