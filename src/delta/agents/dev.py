"""Developer agent.

This agent will handle coding workflows, code review, tests, and refactoring.
"""

from __future__ import annotations

from delta.agents.base import Agent


class DevAgent(Agent):
    """Handles development-centric tasks."""

    name = "dev"

    async def run(self, query: str, source: str, session_id: str | None = None) -> str:
        return "Dev agent scaffold ready. Code intelligence pipeline not implemented yet."
