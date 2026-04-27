"""Central flags for tool execution policy (opt-in automation)."""

from __future__ import annotations

import os


def _truthy(val: str | None) -> bool:
    if val is None:
        return False
    return val.strip().lower() in ("1", "true", "yes", "on")


def deltai_tool_auto_approve() -> bool:
    """
    When True, the LLM may request destructive/privileged tool paths that are
    otherwise rejected (e.g. arch_rollback_plan with apply_etc).

    Default: off. Set DELTAI_TOOL_AUTO_APPROVE=1 in project/.env to opt in.
    """
    return _truthy(os.getenv("DELTAI_TOOL_AUTO_APPROVE"))
