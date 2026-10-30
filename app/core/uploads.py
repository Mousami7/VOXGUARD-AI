from __future__ import annotations

from pathlib import Path
from typing import Optional, Set, Tuple

from fastapi import HTTPException, UploadFile

from app.core.config import settings

READ_CHUNK_SIZE = 64 * 1024


def validate_upload_filename(
    filename: Optional[str],
    allowed_extensions: Set[str],
) -> Tuple[str, str]:
    if not filename or not filename.strip():
        raise HTTPException(status_code=400, detail="A filename is required.")

    safe_name = Path(filename).name
    if not safe_name or safe_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="A filename is required.")

    extension = Path(safe_name).suffix.lower()
    if extension not in allowed_extensions:
        allowed = ", ".join(sorted(allowed_extensions))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension or 'unknown'}. Allowed types: {allowed}",
        )
    return safe_name, extension


async def read_upload_bounded(
    upload: UploadFile,
    max_bytes: Optional[int] = None,
) -> bytes:
    """Read an upload in chunks and reject files over the configured size."""
    limit = max_bytes if max_bytes is not None else settings.max_upload_bytes
    buffer = bytearray()

    while True:
        chunk = await upload.read(READ_CHUNK_SIZE)
        if not chunk:
            break
        if len(buffer) + len(chunk) > limit:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum size of {limit} bytes.",
            )
        buffer.extend(chunk)

    if not buffer:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    return bytes(buffer)
