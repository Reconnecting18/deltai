"""Abstract base class for DELTA agents.

All agents implement a uniform async interface so the orchestrator can route
requests consistently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from delta.config import Settings


class Agent(ABC):
    """Base contract for all DELTA agents."""

    name: str = "agent"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    async def run(self, query: str, source: str, session_id: str | None = None) -> str:
        """Execute the agent task and return a user-facing result string."""
        raise NotImplementedError
