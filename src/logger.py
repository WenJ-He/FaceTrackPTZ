"""Structured JSON logging for FaceTrack PTZ."""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON with required fields."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "state": getattr(record, "state", ""),
            "target_id": getattr(record, "target_id", ""),
            "stage": getattr(record, "stage", ""),
            "event": record.getMessage(),
            "result": getattr(record, "result", ""),
        }
        extra_fields = getattr(record, "extra_data", None)
        if extra_fields:
            log_entry.update(extra_fields)
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Configure and return the application logger with JSON output."""
    logger = logging.getLogger("facetrack")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = JSONFormatter()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log(
    event: str,
    state: str = "",
    target_id: Any = "",
    stage: Any = "",
    result: str = "",
    **kwargs: Any,
) -> None:
    """Emit a structured log entry with the required fields."""
    logger = logging.getLogger("facetrack")
    extra: Dict[str, Any] = {
        "state": state,
        "target_id": target_id,
        "stage": stage,
        "result": result,
    }
    if kwargs:
        extra["extra_data"] = kwargs
    logger.info(event, extra=extra)
