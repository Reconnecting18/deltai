"""
Appwrite bridge — Storage + Functions for cross-machine file handoff and optional cloud execution.

Pairs with server_network: push a single file from a registered Linux host using curl + env file
on that host (~/.config/deltai/appwrite.env), so secrets never cross the SSH command line.

Appwrite Functions are serverless with platform limits; they complement — not replace — HPC or a
dedicated GPU box. Use them for bounded API-style jobs you deploy as functions.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
from typing import Any

from safe_errors import log_exception, public_error_detail

logger = logging.getLogger("deltai.extensions.appwrite_bridge")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "appwrite_status",
            "description": (
                "Report whether Appwrite env vars are set for DeltAI (endpoint, project, API key, "
                "default bucket). Does not call the network. Use before sync or remote push."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "appwrite_storage_list",
            "description": (
                "List files in an Appwrite Storage bucket (pagination). Uses DELTAI_APPWRITE_* env; "
                "bucket_id optional if DELTAI_APPWRITE_BUCKET_ID is set."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bucket_id": {"type": "string", "description": "Override bucket id (optional)."},
                    "limit": {"type": "integer", "description": "Page size 1–100.", "default": 25},
                    "offset": {"type": "integer", "default": 0},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "appwrite_storage_upload",
            "description": (
                "Upload a local file into Appwrite Storage from an allowed DeltAI workspace path "
                "(same roots as read_file). file_id defaults to Appwrite 'unique()'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "local_path": {"type": "string", "description": "File path under workspace roots."},
                    "bucket_id": {"type": "string"},
                    "file_id": {"type": "string", "description": "Optional; default unique()."},
                },
                "required": ["local_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "appwrite_storage_download",
            "description": (
                "Download a Storage file by id to a local path under allowed DeltAI workspace roots."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_id": {"type": "string"},
                    "local_path": {"type": "string"},
                    "bucket_id": {"type": "string"},
                },
                "required": ["file_id", "local_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "appwrite_storage_delete",
            "description": "Delete a file from Appwrite Storage by id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_id": {"type": "string"},
                    "bucket_id": {"type": "string"},
                },
                "required": ["file_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "appwrite_function_execute",
            "description": (
                "Invoke an Appwrite Function by id. "
                "body is the HTTP body string the function receives (often a JSON string). "
                "async_execution true runs in the background. "
                "Note: functions have CPU/time limits — not a full compute cluster."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "function_id": {"type": "string"},
                    "body": {
                        "type": "string",
                        "description": "Execution body string (default {}).",
                        "default": "{}",
                    },
                    "async_execution": {"type": "boolean", "default": False},
                },
                "required": ["function_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "appwrite_remote_storage_push",
            "description": (
                "On a server_network-registered Linux host, upload one file to Appwrite Storage via curl. "
                "Requires that host to have ~/.config/deltai/appwrite.env with DELTAI_APPWRITE_ENDPOINT, "
                "DELTAI_APPWRITE_PROJECT_ID, DELTAI_APPWRITE_API_KEY, DELTAI_APPWRITE_BUCKET_ID. "
                "API key is not passed over SSH; it stays in that file on the remote machine."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "server_id": {"type": "string", "description": "Id from server_network_list."},
                    "remote_file": {
                        "type": "string",
                        "description": "Absolute path on the remote host to a single file.",
                    },
                    "file_id": {
                        "type": "string",
                        "description": "Appwrite fileId (optional; default unique()).",
                    },
                    "timeout_sec": {"type": "integer", "default": 300},
                },
                "required": ["server_id", "remote_file"],
            },
        },
    },
]


def _json_result(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)


def _status_handler(**_kwargs) -> str:
    from . import api as aw

    return _json_result({"ok": True, "config": aw.config_status()})


def _list_handler(
    bucket_id: str | None = None,
    limit: int = 25,
    offset: int = 0,
) -> str:
    from . import api as aw

    try:
        out = aw.storage_list(bucket_id=bucket_id, limit=limit, offset=offset)
        return _json_result({"ok": True, "result": out})
    except Exception as exc:
        log_exception(logger, "appwrite_storage_list failed", exc)
        return _json_result({"ok": False, "error": public_error_detail(exc)})


def _upload_handler(
    local_path: str,
    bucket_id: str | None = None,
    file_id: str | None = None,
) -> str:
    import path_guard
    from . import api as aw

    try:
        resolved = path_guard.resolve_tool_path(local_path)
        if not os.path.isfile(resolved):
            return _json_result({"ok": False, "error": "local_path is not a file"})
        out = aw.storage_upload(local_path=resolved, bucket_id=bucket_id, file_id=file_id)
        return _json_result({"ok": True, "result": out, "local_path": resolved})
    except Exception as exc:
        log_exception(logger, "appwrite_storage_upload failed", exc)
        return _json_result({"ok": False, "error": public_error_detail(exc)})


def _download_handler(
    file_id: str,
    local_path: str,
    bucket_id: str | None = None,
) -> str:
    import path_guard
    from . import api as aw

    try:
        resolved = path_guard.resolve_tool_path(local_path)
        out = aw.storage_download(file_id=file_id, local_path=resolved, bucket_id=bucket_id)
        return _json_result({"ok": True, "result": out})
    except Exception as exc:
        log_exception(logger, "appwrite_storage_download failed", exc)
        return _json_result({"ok": False, "error": public_error_detail(exc)})


def _delete_handler(file_id: str, bucket_id: str | None = None) -> str:
    from . import api as aw

    try:
        out = aw.storage_delete(file_id=file_id, bucket_id=bucket_id)
        return _json_result({"ok": True, "result": out})
    except Exception as exc:
        log_exception(logger, "appwrite_storage_delete failed", exc)
        return _json_result({"ok": False, "error": public_error_detail(exc)})


def _execute_handler(
    function_id: str,
    body: str = "{}",
    async_execution: bool = False,
) -> str:
    from . import api as aw

    try:
        out = aw.function_execute(
            function_id=function_id,
            body=body or "{}",
            async_execution=async_execution,
        )
        return _json_result({"ok": True, "result": out})
    except Exception as exc:
        log_exception(logger, "appwrite_function_execute failed", exc)
        return _json_result({"ok": False, "error": public_error_detail(exc)})


def _bash_single_quoted(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _validate_appwrite_file_id(fid: str) -> str | None:
    f = (fid or "unique()").strip() or "unique()"
    if f == "unique()":
        return f
    if re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,35}", f):
        return f
    return None


def _remote_push_handler(
    server_id: str,
    remote_file: str,
    file_id: str | None = None,
    timeout_sec: int = 300,
) -> str:
    from extensions.server_network import registry as srv_reg

    rf = (remote_file or "").strip()
    if not rf or not os.path.isabs(rf):
        return _json_result(
            {"ok": False, "error": "remote_file must be a non-empty absolute POSIX path"}
        )
    if any(c in rf for c in ("\n", "\r", ";", "|", "&", "`", "$(")):
        return _json_result({"ok": False, "error": "remote_file contains disallowed characters"})
    quoted = shlex.quote(rf)
    fid_ok = _validate_appwrite_file_id((file_id or "unique()").strip() or "unique()")
    if not fid_ok:
        return _json_result(
            {
                "ok": False,
                "error": "file_id must be 'unique()' or a short alphanumeric id "
                "([a-zA-Z0-9._-])",
            }
        )
    fid_bash = _bash_single_quoted(fid_ok)
    # Remote sources env (secrets stay on that host) then curl multipart upload.
    script = f"""set -euo pipefail
ENV_FILE="${{HOME}}/.config/deltai/appwrite.env"
if [ ! -f "$ENV_FILE" ]; then
  echo "missing $ENV_FILE on remote host" >&2
  exit 2
fi
# shellcheck source=/dev/null
. "$ENV_FILE"
: "${{DELTAI_APPWRITE_ENDPOINT:?missing in appwrite.env}}"
: "${{DELTAI_APPWRITE_PROJECT_ID:?missing in appwrite.env}}"
: "${{DELTAI_APPWRITE_API_KEY:?missing in appwrite.env}}"
: "${{DELTAI_APPWRITE_BUCKET_ID:?missing in appwrite.env}}"
FILE_ID={fid_bash}
RF={quoted}
test -f "$RF"
exec curl -sS -X POST \\
  -H "X-Appwrite-Project: $DELTAI_APPWRITE_PROJECT_ID" \\
  -H "X-Appwrite-Key: $DELTAI_APPWRITE_API_KEY" \\
  -F "fileId=$FILE_ID" \\
  -F "file=@${{RF}}" \\
  "$DELTAI_APPWRITE_ENDPOINT/storage/buckets/$DELTAI_APPWRITE_BUCKET_ID/files"
"""

    try:
        out = srv_reg.run_remote_script(server_id, script, timeout_sec=timeout_sec)
        return _json_result({"ok": True, "ssh": out, "hint": "check ssh.stdout for Appwrite JSON"})
    except ValueError as exc:
        log_exception(logger, "appwrite_remote_storage_push validation failed", exc)
        return _json_result({"ok": False, "error": public_error_detail(exc)})
    except Exception as exc:
        log_exception(logger, "appwrite_remote_storage_push failed", exc)
        return _json_result({"ok": False, "error": public_error_detail(exc)})


def setup(app) -> None:
    from fastapi import APIRouter
    from tools.executor import register_handler

    register_handler("appwrite_status", _status_handler)
    register_handler("appwrite_storage_list", _list_handler)
    register_handler("appwrite_storage_upload", _upload_handler)
    register_handler("appwrite_storage_download", _download_handler)
    register_handler("appwrite_storage_delete", _delete_handler)
    register_handler("appwrite_function_execute", _execute_handler)
    register_handler("appwrite_remote_storage_push", _remote_push_handler)

    router = APIRouter(prefix="/ext/appwrite_bridge", tags=["appwrite_bridge"])

    @router.get("/status")
    def http_status():
        from . import api as aw

        return {"ok": True, "config": aw.config_status()}

    @router.get("/remote-env.example")
    def remote_env_example():
        """Example lines for ~/.config/deltai/appwrite.env on a Linux workstation."""
        lines = [
            "DELTAI_APPWRITE_ENDPOINT=https://cloud.appwrite.io/v1",
            "DELTAI_APPWRITE_PROJECT_ID=",
            "DELTAI_APPWRITE_API_KEY=",
            "DELTAI_APPWRITE_BUCKET_ID=",
            "export DELTAI_APPWRITE_ENDPOINT DELTAI_APPWRITE_PROJECT_ID",
            "export DELTAI_APPWRITE_API_KEY DELTAI_APPWRITE_BUCKET_ID",
        ]
        return {"path": "~/.config/deltai/appwrite.env", "lines": lines}

    app.include_router(router)
    logger.info("appwrite_bridge: tools + routes at /ext/appwrite_bridge/")


def shutdown() -> None:
    logger.info("appwrite_bridge: shutdown")
