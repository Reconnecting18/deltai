"""Context and memory orchestration helpers.

This module coordinates short-term context assembly for each request and
persisted memory lookup from SQLite.
"""

from __future__ import annotations

from dataclasses import dataclass

from delta.platform.context_capture import DesktopContext, capture_context


@dataclass
class RequestContext:
    """Aggregated context passed into agent execution."""

    source: str
    session_id: str | None
    desktop: DesktopContext


def build_request_context(source: str, session_id: str | None) -> RequestContext:
    """Create a standardized context object for one request."""
    return RequestContext(source=source, session_id=session_id, desktop=capture_context())
