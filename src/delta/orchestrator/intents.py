"""Intent parsing helpers for DELTA.

Maps natural-language requests into coarse intent categories so the
orchestrator can select the correct agent.
"""

from __future__ import annotations



def classify_intent(query: str) -> str:
    """Return a coarse intent label used for initial agent routing."""
    q = query.lower()
    if any(token in q for token in ("bash", "shell", "terminal", "command")):
        return "shell"
    if any(token in q for token in ("file", "directory", "folder", "path")):
        return "file"
    if any(token in q for token in ("code", "test", "refactor", "debug")):
        return "dev"
    if any(token in q for token in ("context", "clipboard", "window", "active app")):
        return "context"
    if any(token in q for token in ("workflow", "plan", "pipeline", "automate")):
        return "workflow"
    return "general"
