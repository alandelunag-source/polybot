"""
Structured logging configuration for polybot.
Outputs JSON-formatted logs to stdout and optionally to a file.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Optional


class JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure the root logger.

    Args:
        level:    Log level (e.g. logging.DEBUG)
        log_file: Optional path to write logs to in addition to stdout
    """
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(JsonFormatter())
    root.addHandler(console)

    # Optional file handler
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(JsonFormatter())
        root.addHandler(fh)

    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("web3").setLevel(logging.WARNING)
