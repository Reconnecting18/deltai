"""File agent.

This agent will perform file system reads/writes/search with safety guards.
"""

from __future__ import annotations

from delta.agents.base import Agent


class FileAgent(Agent):
    """Handles file navigation and editing tasks."""

    name = "file"

    async def run(self, query: str, source: str, session_id: str | None = None) -> str:
        return "File agent scaffold ready. File operation policies not implemented yet."
