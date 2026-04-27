"""
Model Context Protocol bridge: expose the merged Ollama-style tool catalog via MCP.

Uses the low-level mcp Server so tool definitions keep their JSON Schema from
tools/definitions.py without per-tool Python stubs.

Optional dependency: pip install -e ".[mcp]" from the repo root.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger("deltai.mcp")

_catalog_ready = False
_mcp_server_singleton: Any = None
_http_session_manager: Any = None

MCP_INSTRUCTIONS = (
    "deltai MCP bridge: same tool catalog as POST /chat on the dev server. "
    "Tools run on the host as the current OS user (user-space; not a sandbox). "
    "Prefer read-only tools unless the operator requested changes."
)


def ensure_mcp_tool_catalog() -> None:
    """
    Load project/.env, discover extensions, merge extension tools into TOOLS — same
    order as main.py. Idempotent per process.
    """
    global _catalog_ready
    if _catalog_ready:
        return

    from dotenv import load_dotenv

    load_dotenv()

    from fastapi import FastAPI

    from extensions import get_extension_tools, load_extensions
    from tools.definitions import TOOLS, _merge_extension_tools

    stub = FastAPI()
    load_extensions(stub)
    _merge_extension_tools(get_extension_tools())
    _catalog_ready = True
    logger.info("MCP: tool catalog ready (%d tools)", len(TOOLS))


def ollama_catalog_to_mcp_tools():
    """Build mcp.types.Tool list from the global TOOLS list."""
    import mcp.types as mtypes

    from tools.definitions import TOOLS

    out: list[mtypes.Tool] = []
    for entry in TOOLS:
        fn = entry.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        schema = fn.get("parameters") or {"type": "object", "properties": {}}
        out.append(
            mtypes.Tool(
                name=name,
                description=fn.get("description") or "",
                inputSchema=schema,
            )
        )
    return out


def build_mcp_server():
    """
    Create a low-level MCP Server with list_tools / call_tool wired to execute_tool.
    Safe to call once per process; handlers read the live TOOLS list on each request.
    """
    import mcp.types as mtypes
    from mcp.server.lowlevel import Server

    from tools.executor import execute_tool

    server = Server(
        "deltai",
        version="0.1.0",
        instructions=MCP_INSTRUCTIONS,
    )

    @server.list_tools()
    async def _list_tools() -> list[mtypes.Tool]:
        return ollama_catalog_to_mcp_tools()

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any] | None):
        args = arguments or {}
        try:
            result = await asyncio.to_thread(execute_tool, name, args)
            text = str(result)
            is_err = text.startswith("ERROR:")
            return mtypes.CallToolResult(
                content=[mtypes.TextContent(type="text", text=text)],
                isError=is_err,
            )
        except Exception as exc:
            logger.exception("MCP tool %s failed", name)
            return mtypes.CallToolResult(
                content=[mtypes.TextContent(type="text", text=str(exc))],
                isError=True,
            )

    return server


def get_mcp_server_singleton():
    """Shared Server instance for HTTP mount (one StreamableHTTPSessionManager per app)."""
    global _mcp_server_singleton
    if _mcp_server_singleton is None:
        _mcp_server_singleton = build_mcp_server()
    return _mcp_server_singleton


async def run_stdio_async() -> None:
    """Run MCP over stdin/stdout (for IDE / CLI clients)."""
    from mcp.server.stdio import stdio_server

    ensure_mcp_tool_catalog()
    server = build_mcp_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def get_streamable_http_asgi_app(server=None):
    """
    Stateless Streamable HTTP ASGI app for mounting on FastAPI/Starlette.
    One StreamableHTTPSessionManager per process.
    """
    from mcp.server.streamable_http import StreamableHTTPASGIApp
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    global _http_session_manager
    srv = server or get_mcp_server_singleton()
    if _http_session_manager is None:
        _http_session_manager = StreamableHTTPSessionManager(
            app=srv,
            event_store=None,
            json_response=False,
            stateless=True,
            security_settings=None,
            retry_interval=None,
        )
    return StreamableHTTPASGIApp(_http_session_manager)
