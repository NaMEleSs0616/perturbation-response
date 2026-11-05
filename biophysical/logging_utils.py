"""Structured (JSON-lines capable) logging."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "data", None)
        if extra:
            payload["data"] = extra
        return json.dumps(payload, default=str)


def get_logger(name: str = "biophysical") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger("biophysical").handlers:
        root = logging.getLogger("biophysical")
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    return logger


def add_json_file_handler(path: str | Path) -> logging.Handler:
    """Attach a JSON-lines log file for a run; returns the handler so callers can remove it."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path)
    handler.setFormatter(_JsonFormatter())
    logging.getLogger("biophysical").addHandler(handler)
    return handler
