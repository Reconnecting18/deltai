"""Desktop context capture abstraction.

This module will provide active-window and clipboard context capture for both
Wayland and X11 sessions through pluggable providers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DesktopContext:
    """Captured desktop context used by ContextAgent and Orchestrator."""

    active_window_title: str = ""
    active_window_app: str = ""
    clipboard_text: str = ""
    display_server: str = "unknown"


def capture_context() -> DesktopContext:
    """Capture context from current desktop session.

    Placeholder implementation until Wayland/X11 backends are added.
    """
    return DesktopContext()
