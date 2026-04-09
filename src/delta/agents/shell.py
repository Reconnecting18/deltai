"""Shell agent.

This agent will execute and supervise Linux shell tasks under policy controls.
"""

from __future__ import annotations

from delta.agents.base import Agent


class ShellAgent(Agent):
    """Handles shell and command-line task execution requests."""

    name = "shell"

    async def run(self, query: str, source: str, session_id: str | None = None) -> str:
        return "Shell agent scaffold ready. Command execution policy not implemented yet."
