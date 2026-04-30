"""
Server network extension — inventory of local Linux servers and bounded SSH automation.

Stores a small JSON registry under DELTA_DATA_DIR (see registry.registry_path).
The model can list/add/update/remove servers and run commands only on registered hosts
(no arbitrary ssh targets). Uses BatchMode=yes (non-interactive; keys or ssh-agent).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel
from safe_errors import public_error_detail


class ServerAdd(BaseModel):
    host: str
    user: str
    port: int = 22
    label: str | None = None
    tags: list[str] | None = None
    notes: str | None = None
    identity_file: str | None = None


class ServerUpdate(BaseModel):
    label: str | None = None
    tags: list[str] | None = None
    notes: str | None = None
    identity_file: str | None = None
    clear_identity_file: bool = False


class RunCommandBody(BaseModel):
    command: str
    timeout_sec: int = 120


class ScriptBody(BaseModel):
    script: str
    timeout_sec: int = 300

logger = logging.getLogger("deltai.extensions.server_network")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "server_network_list",
            "description": (
                "List registered Linux servers in the local server network inventory "
                "(host, ssh user, port, tags, notes). Use for automation planning and status."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "server_network_add",
            "description": (
                "Register a server reachable via SSH (user-space). "
                "Requires non-interactive auth (ssh-agent or key under home). "
                "Duplicate host+port+user is rejected."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "host": {"type": "string", "description": "Hostname or IP."},
                    "user": {"type": "string", "description": "SSH login user."},
                    "port": {
                        "type": "integer",
                        "description": "SSH port (default 22).",
                        "default": 22,
                    },
                    "label": {"type": "string", "description": "Short display name (optional)."},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional tags (e.g. role, site).",
                    },
                    "notes": {"type": "string", "description": "Free-form notes (optional)."},
                    "identity_file": {
                        "type": "string",
                        "description": "Path to private key under $HOME (optional).",
                    },
                },
                "required": ["host", "user"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "server_network_update",
            "description": "Update label, tags, notes, or identity file for a registered server.",
            "parameters": {
                "type": "object",
                "properties": {
                    "server_id": {"type": "string", "description": "Server id from list/add."},
                    "label": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "string"},
                    "identity_file": {"type": "string"},
                    "clear_identity_file": {
                        "type": "boolean",
                        "description": "If true, remove custom identity file.",
                        "default": False,
                    },
                },
                "required": ["server_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "server_network_remove",
            "description": "Remove a server from the inventory by id.",
            "parameters": {
                "type": "object",
                "properties": {"server_id": {"type": "string"}},
                "required": ["server_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "server_network_probe",
            "description": (
                "Check SSH connectivity to a registered server (runs ssh … true). "
                "Use after add or when diagnosing automation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "server_id": {"type": "string"},
                    "timeout_sec": {"type": "integer", "default": 8},
                },
                "required": ["server_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "server_network_run_command",
            "description": (
                "Run a single shell command on a registered server via SSH. "
                "Only works for servers in the inventory; no arbitrary hosts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "server_id": {"type": "string"},
                    "command": {"type": "string", "description": "Remote command line."},
                    "timeout_sec": {"type": "integer", "default": 120},
                },
                "required": ["server_id", "command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "server_network_run_script",
            "description": (
                "Run a multi-line bash script on a registered server (stdin to bash -s). "
                "Use for small automation snippets; respect operator policies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "server_id": {"type": "string"},
                    "script": {"type": "string", "description": "Script body passed to bash -s."},
                    "timeout_sec": {"type": "integer", "default": 300},
                },
                "required": ["server_id", "script"],
            },
        },
    },
]


def _json_result(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)


def _list_handler(**_kwargs) -> str:
    from . import registry as reg

    return _json_result({"registry_path": reg.registry_path(), "servers": reg.list_servers()})


def _add_handler(
    host: str,
    user: str,
    port: int = 22,
    label: str | None = None,
    tags: list | None = None,
    notes: str | None = None,
    identity_file: str | None = None,
) -> str:
    from . import registry as reg

    try:
        rec = reg.add_server(
            host=host,
            user=user,
            port=port,
            label=label,
            tags=tags,
            notes=notes,
            identity_file=identity_file,
        )
        return _json_result({"ok": True, "server": rec})
    except ValueError as exc:
        return _json_result({"ok": False, "error": public_error_detail(exc)})


def _update_handler(
    server_id: str,
    label: str | None = None,
    tags: list | None = None,
    notes: str | None = None,
    identity_file: str | None = None,
    clear_identity_file: bool = False,
) -> str:
    from . import registry as reg

    try:
        rec = reg.update_server(
            server_id=server_id,
            label=label,
            tags=tags,
            notes=notes,
            identity_file=identity_file,
            clear_identity_file=clear_identity_file,
        )
        return _json_result({"ok": True, "server": rec})
    except ValueError as exc:
        return _json_result({"ok": False, "error": public_error_detail(exc)})


def _remove_handler(server_id: str) -> str:
    from . import registry as reg

    ok = reg.remove_server(server_id)
    return _json_result({"ok": ok})


def _probe_handler(server_id: str, timeout_sec: int = 8) -> str:
    from . import registry as reg

    try:
        out = reg.probe_server(server_id, timeout_sec=timeout_sec)
        return _json_result(out)
    except ValueError as exc:
        return _json_result({"ok": False, "error": public_error_detail(exc)})


def _run_cmd_handler(server_id: str, command: str, timeout_sec: int = 120) -> str:
    from . import registry as reg

    try:
        out = reg.run_remote_command(server_id, command, timeout_sec=timeout_sec)
        return _json_result(out)
    except ValueError as exc:
        return _json_result({"ok": False, "error": public_error_detail(exc)})


def _run_script_handler(server_id: str, script: str, timeout_sec: int = 300) -> str:
    from . import registry as reg

    try:
        out = reg.run_remote_script(server_id, script, timeout_sec=timeout_sec)
        return _json_result(out)
    except ValueError as exc:
        return _json_result({"ok": False, "error": public_error_detail(exc)})


def setup(app) -> None:
    from fastapi import APIRouter
    from tools.executor import register_handler

    register_handler("server_network_list", _list_handler)
    register_handler("server_network_add", _add_handler)
    register_handler("server_network_update", _update_handler)
    register_handler("server_network_remove", _remove_handler)
    register_handler("server_network_probe", _probe_handler)
    register_handler("server_network_run_command", _run_cmd_handler)
    register_handler("server_network_run_script", _run_script_handler)

    from . import registry as reg

    router = APIRouter(prefix="/ext/server_network", tags=["server_network"])

    @router.get("/registry-path")
    def registry_path_endpoint():
        return {"path": reg.registry_path()}

    @router.get("/servers")
    def http_list_servers():
        return {"servers": reg.list_servers()}

    @router.post("/servers")
    def http_add_server(body: ServerAdd):
        try:
            return {"ok": True, "server": reg.add_server(**body.model_dump())}
        except ValueError as exc:
            return {"ok": False, "error": public_error_detail(exc)}

    @router.patch("/servers/{server_id}")
    def http_update_server(server_id: str, body: ServerUpdate):
        try:
            d = body.model_dump(exclude_unset=True)
            d["server_id"] = server_id
            return {"ok": True, "server": reg.update_server(**d)}
        except ValueError as exc:
            return {"ok": False, "error": public_error_detail(exc)}

    @router.delete("/servers/{server_id}")
    def http_remove_server(server_id: str):
        return {"ok": reg.remove_server(server_id)}

    @router.post("/servers/{server_id}/probe")
    def http_probe(server_id: str, timeout_sec: int = 8):
        return reg.probe_server(server_id, timeout_sec=timeout_sec)

    @router.post("/servers/{server_id}/run")
    def http_run(server_id: str, body: RunCommandBody):
        return reg.run_remote_command(server_id, body.command, timeout_sec=body.timeout_sec)

    @router.post("/servers/{server_id}/run-script")
    def http_run_script(server_id: str, body: ScriptBody):
        return reg.run_remote_script(server_id, body.script, timeout_sec=body.timeout_sec)

    app.include_router(router)
    logger.info("server_network: tools + routes at /ext/server_network/")


def shutdown() -> None:
    logger.info("server_network: shutdown")
