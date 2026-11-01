from __future__ import annotations

import time
from typing import Any, Dict, List

import httpx

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("llm")

REFUSAL_MESSAGE = (
    "I could not find enough evidence in the uploaded documents to answer that question."
)

SYSTEM_PROMPT = """You are VoxGuard AI. Answer using only the numbered evidence passages.
Rules:
- Use only facts that appear in the passages.
- If the passages are not enough, reply with exactly this sentence: I could not find enough evidence in the uploaded documents to answer that question.
- After a successful answer, add a final line starting with Sources: and list the passage numbers you used, such as Sources: [1], [2].
- Do not invent names, dates, policies, or numbers.
"""


def is_llm_configured() -> bool:
    return settings.llm_enabled()


def generate_grounded_answer(question: str, sources: List[Dict[str, Any]]) -> str:
    """Generate a short answer strictly from retrieved passages."""
    if not is_llm_configured():
        raise RuntimeError("LLM is not configured")

    evidence_blocks = []
    for index, source in enumerate(sources, start=1):
        evidence_blocks.append(
            "[{index}] source={source_name}\n{text}".format(
                index=index,
                source_name=source.get("source") or "document",
                text=source.get("text") or "",
            )
        )

    user_prompt = "Question: {question}\n\nEvidence:\n{evidence}".format(
        question=question,
        evidence="\n\n".join(evidence_blocks),
    )

    started = time.perf_counter()
    url = settings.openai_base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": "Bearer {key}".format(key=settings.openai_api_key),
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.openai_model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()
        answer = (
            body.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        if not answer:
            raise RuntimeError("LLM returned an empty answer")
        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.info(
            "llm_completed error_category=none model=%s duration_ms=%s",
            settings.openai_model,
            duration_ms,
        )
        return answer
    except Exception:
        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.exception(
            "llm_failed error_category=llm duration_ms=%s",
            duration_ms,
        )
        raise
