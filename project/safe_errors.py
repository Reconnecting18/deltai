"""Avoid leaking exception strings to clients or tool output (CodeQL py/stack-trace-exposure)."""

from __future__ import annotations

import json
import logging


def log_exception(logger: logging.Logger, message: str, exc: BaseException) -> None:
    """Log full traceback server-side."""
    logger.error("%s", message, exc_info=(type(exc), exc, exc.__traceback__))


def public_error_detail(
    exc: BaseException,
    *,
    generic: str = "An unexpected error occurred",
) -> str:
    """
    Return a client-safe error string. Known benign validation errors may pass through.
    """
    if isinstance(exc, (ValueError, KeyError, TypeError, json.JSONDecodeError)):
        return str(exc)
    return generic
