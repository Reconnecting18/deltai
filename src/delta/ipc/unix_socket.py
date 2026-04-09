"""Line-delimited JSON Unix socket IPC server.

This server provides lightweight command dispatch for local interfaces that
prefer raw socket communication over HTTP.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any


class IPCServer:
    """Async Unix socket server for local DELTA interface communication."""

    def __init__(self, socket_path: str, orchestrator: Any) -> None:
        self.socket_path = socket_path
        self.orchestrator = orchestrator
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        """Create and start the Unix socket server."""
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
        self._server = await asyncio.start_unix_server(self._handle_client, path=self.socket_path)

    async def stop(self) -> None:
        """Stop server and remove the socket file."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Process newline-delimited JSON requests from one client."""
        try:
            line = await reader.readline()
            payload = json.loads(line.decode("utf-8") or "{}")
            result = await self.orchestrator.handle_request(
                query=payload.get("query", ""),
                source=payload.get("source", "ipc"),
                session_id=payload.get("session_id"),
            )
            writer.write((json.dumps(result) + "\n").encode("utf-8"))
            await writer.drain()
        except Exception as exc:
            writer.write((json.dumps({"status": "error", "output": str(exc), "agent": "none"}) + "\n").encode("utf-8"))
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
