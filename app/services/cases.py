from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("cases")


def _store_path():
    settings.cases_dir.mkdir(parents=True, exist_ok=True)
    return settings.cases_dir / "cases.json"


def _load_store() -> Dict[str, Any]:
    path = _store_path()
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.exception("case_store_invalid error_category=internal")
        return {}
    if not isinstance(loaded, dict):
        return {}
    return loaded


def _save_store(store: Dict[str, Any]) -> None:
    path = _store_path()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(store, indent=2), encoding="utf-8")
    temporary.replace(path)


def create_case(title: str) -> Dict[str, Any]:
    store = _load_store()
    case_id = str(uuid.uuid4())
    case = {
        "id": case_id,
        "title": title.strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "audio_filename": None,
        "transcript": None,
        "transcript_language": None,
        "documents": [],
    }
    store[case_id] = case
    _save_store(store)
    logger.info("case_created error_category=none case_id=%s", case_id)
    return case


def get_case(case_id: str) -> Dict[str, Any]:
    store = _load_store()
    if case_id not in store:
        raise KeyError(case_id)
    return store[case_id]


def list_cases() -> List[Dict[str, Any]]:
    cases = list(_load_store().values())
    return sorted(cases, key=lambda item: item.get("created_at") or "", reverse=True)


def update_case(case_id: str, **changes: Any) -> Dict[str, Any]:
    store = _load_store()
    if case_id not in store:
        raise KeyError(case_id)
    store[case_id].update(changes)
    _save_store(store)
    return store[case_id]


def summarize_case(case: Dict[str, Any]) -> Dict[str, Any]:
    documents = case.get("documents") or []
    return {
        "id": case["id"],
        "title": case["title"],
        "created_at": case["created_at"],
        "audio_filename": case.get("audio_filename"),
        "document_count": len(documents),
        "has_transcript": bool(case.get("transcript")),
    }
