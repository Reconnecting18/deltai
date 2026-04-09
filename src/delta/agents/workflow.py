"""Workflow agent.

This agent coordinates multi-step plans and delegates work to specialist agents.
"""

from __future__ import annotations

from delta.agents.base import Agent


class WorkflowAgent(Agent):
    """Default coordinator for general or mixed-intent requests."""

    name = "workflow"

    async def run(self, query: str, source: str, session_id: str | None = None) -> str:
        return "Workflow agent scaffold ready. Task decomposition not implemented yet."
