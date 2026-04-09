"""Central request orchestrator for DELTA.

The orchestrator is responsible for:
- Intent classification
- Agent selection and dispatch
- Context assembly and memory interaction
- Fallback model behavior selection
"""

from __future__ import annotations

from typing import Any

from delta.agents.context import ContextAgent
from delta.agents.dev import DevAgent
from delta.agents.file import FileAgent
from delta.agents.shell import ShellAgent
from delta.agents.workflow import WorkflowAgent
from delta.config import Settings
from delta.orchestrator.context import build_request_context
from delta.orchestrator.intents import classify_intent


class Orchestrator:
    """Top-level coordinator for all DELTA task execution."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._agents: dict[str, Any] = {
            "shell": ShellAgent(settings=settings),
            "file": FileAgent(settings=settings),
            "dev": DevAgent(settings=settings),
            "context": ContextAgent(settings=settings),
            "workflow": WorkflowAgent(settings=settings),
        }

    async def handle_request(
        self,
        query: str,
        source: str,
        session_id: str | None = None,
    ) -> dict[str, str]:
        """Classify an intent and route it to an agent."""
        if not query.strip():
            return {"status": "error", "output": "Query is empty", "agent": "none"}

        _ = build_request_context(source=source, session_id=session_id)
        intent = classify_intent(query)
        agent = self._agents.get(intent) or self._agents["workflow"]
        output = await agent.run(query=query, source=source, session_id=session_id)
        return {"status": "ok", "output": output, "agent": agent.name}
