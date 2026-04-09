"""Context agent.

This agent will gather system context (active window, clipboard, environment)
and normalize it for other agents.
"""

from __future__ import annotations

from delta.agents.base import Agent


class ContextAgent(Agent):
    """Handles contextual data gathering and summarization."""

    name = "context"

    async def run(self, query: str, source: str, session_id: str | None = None) -> str:
        return "Context agent scaffold ready. Wayland/X11 capture not implemented yet."
