"""
Centralized logging configuration for the RAG backend.

Provides structured JSON logging by default with an optional plain-text
formatter for local development. Use :func:`get_logger` to obtain module
loggers so configuration is applied consistently across the project.
"""

from __future__ import annotations

import json
import logging
import logging.config
import os
import threading
from typing import Any, Dict

_CONFIG_LOCK = threading.Lock()
_IS_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """Lightweight JSON formatter without extra dependencies."""

    default_time_format = "%Y-%m-%dT%H:%M:%S"
    default_msec_format = "%s.%03d"

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.default_time_format),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        # Include arbitrary extra attributes that do not belong to logging internals.
        for key, value in record.__dict__.items():
            if key in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
            }:
                continue
            payload[key] = value

        return json.dumps(payload, default=str, ensure_ascii=False)


def _build_logging_config(level: str, formatter: str) -> Dict[str, Any]:
    formatters: Dict[str, Any] = {
        "json": {"()": JsonFormatter},
        "console": {
            "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    }

    handlers: Dict[str, Any] = {
        "default": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": formatter if formatter in formatters else "json",
        }
    }

    loggers: Dict[str, Any] = {
        "": {"handlers": ["default"], "level": level, "propagate": False},
        "uvicorn": {"handlers": ["default"], "level": level, "propagate": False},
        "uvicorn.error": {"handlers": ["default"], "level": level, "propagate": False},
        "uvicorn.access": {"handlers": ["default"], "level": level, "propagate": False},
    }

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": handlers,
        "loggers": loggers,
    }


def configure_logging(force: bool = False) -> None:
    """Apply the shared logging configuration once per process."""
    global _IS_CONFIGURED
    with _CONFIG_LOCK:
        if _IS_CONFIGURED and not force:
            return

        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        log_format = os.getenv("LOG_FORMAT", "json").lower()

        logging.config.dictConfig(_build_logging_config(log_level, log_format))
        _IS_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger that uses the shared configuration."""
    configure_logging()
    return logging.getLogger(name)


