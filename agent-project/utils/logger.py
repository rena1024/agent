"""Structured logger with trace id support and safe extra rendering."""

import json
import logging
import sys
import uuid
from typing import Any, Dict
from colorama import Fore


class ExtraJSONFormatter(logging.Formatter):
    """Formatter that renders non-standard record attrs as JSON in extra_json."""

    DEFAULT_KEYS = {
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
    }

    def format(self, record: logging.LogRecord) -> str:
        extras: Dict[str, Any] = {
            k: v for k, v in record.__dict__.items() if k not in self.DEFAULT_KEYS
        }
        record.extra_json = json.dumps(extras, ensure_ascii=False, default=str)
        return super().format(record)


def get_logger():
    logger = logging.getLogger("agent")
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    formatter = ExtraJSONFormatter(
        Fore.MAGENTA
        + "%(asctime)s %(levelname)s %(name)s %(message)s\n"
        + Fore.YELLOW
        + "extras=%(extra_json)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def new_trace_id() -> str:
    return uuid.uuid4().hex


# convenience to attach to logger instance
logging.Logger.new_trace_id = staticmethod(new_trace_id)  # type: ignore[attr-defined]
