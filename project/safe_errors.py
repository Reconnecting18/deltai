"""Avoid leaking exception strings to clients or tool output (CodeQL py/stack-trace-exposure)."""

from __future__ import annotations

import json
import logging


def log_exception(logger: logging.Logger, message: str, exc: BaseException) -> None:
    """Log exception category without embedding exception text (CodeQL py/log-injection)."""
    logger.error("%s [%s]", message, type(exc).__name__, exc_info=False)


def public_error_detail(
    exc: BaseException,
    *,
    generic: str = "An unexpected error occurred",
) -> str:
    """
    Return a client-safe error string for HTTP/JSON surfaces.

    Does not stringify the exception (avoids CodeQL py/stack-trace-exposure): no
    data flow from exception text to clients; use log_exception for diagnostics.
    """
    if isinstance(exc, ValueError):
        return "Invalid value"
    if isinstance(exc, KeyError):
        return "Missing or invalid key"
    if isinstance(exc, TypeError):
        return "Invalid type"
    if isinstance(exc, json.JSONDecodeError):
        return "Invalid JSON"
    return generic
