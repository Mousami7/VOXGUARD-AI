from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Optional

from app.core.config import settings

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")

LOGGER_NAME = "voxguard"


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get("-")
        return True


def setup_logging() -> logging.Logger:
    """Configure the application logger once."""
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    level_name = (settings.log_level or "INFO").upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logger.level)
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s request_id=%(request_id)s logger=%(name)s %(message)s"
        )
    )
    logger.addHandler(handler)
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)
