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
from delta.storage.reports import write_ai_report


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
        s = self.settings
        if not query.strip():
            result = {"status": "error", "output": "Query is empty", "agent": "none"}
            write_ai_report(
                reports_dir=s.reports_dir,
                enabled=s.ai_reports_enabled,
                category="orchestrator",
                status="error",
                fields={
                    "query": query,
                    "output": result["output"],
                    "agent": result["agent"],
                    "intent": None,
                    "session_id": session_id,
                    "request_source": source,
                    "error": {"detail": "empty_query"},
                },
            )
            return result

        _ = build_request_context(source=source, session_id=session_id)
        intent = classify_intent(query)
        agent = self._agents.get(intent) or self._agents["workflow"]
        try:
            output = await agent.run(query=query, source=source, session_id=session_id)
        except Exception as exc:
            write_ai_report(
                reports_dir=s.reports_dir,
                enabled=s.ai_reports_enabled,
                category="orchestrator",
                status="error",
                fields={
                    "query": query,
                    "output": str(exc),
                    "agent": agent.name,
                    "intent": intent,
                    "session_id": session_id,
                    "request_source": source,
                    "error": {"detail": type(exc).__name__},
                },
            )
            return {"status": "error", "output": str(exc), "agent": agent.name}

        write_ai_report(
            reports_dir=s.reports_dir,
            enabled=s.ai_reports_enabled,
            category="orchestrator",
            status="ok",
            fields={
                "query": query,
                "output": output,
                "agent": agent.name,
                "intent": intent,
                "session_id": session_id,
                "request_source": source,
            },
        )
        return {"status": "ok", "output": output, "agent": agent.name}
