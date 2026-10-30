from __future__ import annotations

import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.core.logging import get_logger, request_id_ctx

logger = get_logger("http")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        token = request_id_ctx.set(request_id)
        started = time.perf_counter()
        route_name = request.scope.get("path", request.url.path)
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((time.perf_counter() - started) * 1000, 1)
            logger.exception(
                "request_failed method=%s route=%s error_category=internal duration_ms=%s",
                request.method,
                route_name,
                duration_ms,
            )
            request_id_ctx.reset(token)
            raise

        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.info(
            "request_completed method=%s route=%s status_code=%s duration_ms=%s",
            request.method,
            route_name,
            response.status_code,
            duration_ms,
        )
        response.headers["X-Request-ID"] = request_id
        request_id_ctx.reset(token)
        return response
