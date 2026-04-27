"""
Optional Streamable HTTP transport for MCP, mounted on the FastAPI app.

Enable with DELTAI_MCP_HTTP_ENABLE=true. Optional DELTAI_MCP_HTTP_KEY requires
Authorization: Bearer <key> (or X-Deltai-Mcp-Key for simple clients).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger("deltai.mcp")


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


class _McpHttpAuthASGI:
    """ASGI wrapper: optional shared secret for mounted MCP HTTP."""

    def __init__(self, app: ASGIApp, key: str | None):
        self.app = app
        self.key = key

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self.key:
            await self.app(scope, receive, send)
            return
        hdrs = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        auth = hdrs.get("authorization", "")
        bearer_ok = auth == f"Bearer {self.key}"
        header_ok = hdrs.get("x-deltai-mcp-key", "") == self.key
        if not bearer_ok and not header_ok:
            await JSONResponse(
                {"detail": "MCP HTTP: send Authorization: Bearer <DELTAI_MCP_HTTP_KEY> or X-Deltai-Mcp-Key"},
                status_code=401,
            )(scope, receive, send)
            return
        await self.app(scope, receive, send)


def maybe_mount_mcp_http(app: Any) -> None:
    """
    Mount MCP Streamable HTTP if DELTAI_MCP_HTTP_ENABLE is set.
    Uses the same merged TOOLS as the rest of the process — call only after
    load_extensions + _merge_extension_tools in main.py.
    """
    if not _env_bool("DELTAI_MCP_HTTP_ENABLE"):
        return

    try:
        from mcp_bridge import get_mcp_server_singleton, get_streamable_http_asgi_app
    except ImportError as e:
        logger.warning("MCP HTTP requested but mcp_bridge import failed [%s]", e)
        return

    path = os.getenv("DELTAI_MCP_HTTP_PATH", "/mcp").strip() or "/mcp"
    if not path.startswith("/"):
        path = "/" + path

    key = os.getenv("DELTAI_MCP_HTTP_KEY", "").strip() or None

    try:
        inner = get_streamable_http_asgi_app(get_mcp_server_singleton())
    except ImportError:
        logger.warning(
            "MCP HTTP disabled: package 'mcp' not installed. Use: pip install -e \".[mcp]\""
        )
        return
    except Exception as e:
        logger.warning("MCP HTTP mount failed [%s]", e)
        return

    wrapped: ASGIApp = _McpHttpAuthASGI(inner, key) if key else inner
    app.mount(path, wrapped)
    logger.info("MCP Streamable HTTP mounted at %s (auth=%s)", path, "on" if key else "off")
