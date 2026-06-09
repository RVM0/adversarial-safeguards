"""Rich-formatted logging for the framework.

Use `get_logger(__name__)` in every module. The first call initializes
the root logger with a Rich handler for human-readable console output
and a JSON file handler if `ADVSAFE_LOG_DIR` is set.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.logging import RichHandler

_INITIALIZED = False


class _JsonFormatter(logging.Formatter):
    """Structured JSON-line formatter for the file handler.

    One line per record; trivially parsed for run analysis.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Attach any extra fields the caller passed via `extra=...`.
        for key, val in record.__dict__.items():
            if (
                key in payload
                or key.startswith("_")
                or key
                in {
                    "args",
                    "asctime",
                    "created",
                    "exc_info",
                    "exc_text",
                    "filename",
                    "funcName",
                    "levelname",
                    "levelno",
                    "lineno",
                    "module",
                    "msecs",
                    "message",
                    "msg",
                    "name",
                    "pathname",
                    "process",
                    "processName",
                    "relativeCreated",
                    "stack_info",
                    "thread",
                    "threadName",
                }
            ):
                continue
            try:
                json.dumps(val)
                payload[key] = val
            except (TypeError, ValueError):
                payload[key] = repr(val)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO", log_dir: str | os.PathLike | None = None) -> None:
    """Configure the root logger once per process."""
    global _INITIALIZED
    if _INITIALIZED:
        return

    root = logging.getLogger()
    root.setLevel(level.upper())

    # Console: Rich
    console = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_path=False,
        markup=False,
    )
    console.setLevel(level.upper())
    console.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    root.addHandler(console)

    # File: JSON-line, only if log_dir set (or env var)
    log_dir = log_dir or os.environ.get("ADVSAFE_LOG_DIR")
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        file_handler = logging.FileHandler(log_path / f"advsafe-{stamp}.jsonl")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(_JsonFormatter())
        root.addHandler(file_handler)

    # Quiet down some noisy libraries
    for noisy in ("urllib3", "filelock", "huggingface_hub", "datasets", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, initializing the root if needed."""
    if not _INITIALIZED:
        setup_logging()
    return logging.getLogger(name)


# Convenience: install a hook so uncaught exceptions are logged structured.
def _excepthook(exc_type, exc_value, exc_tb):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    get_logger("advsafe.uncaught").critical(
        "Uncaught exception", exc_info=(exc_type, exc_value, exc_tb)
    )


sys.excepthook = _excepthook
