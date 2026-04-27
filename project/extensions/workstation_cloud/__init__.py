"""
Workstation cloud sync — thin extension for status, health, and safe tooling.

See docs/linux-workstation-cloud-architecture.md for the full architecture.
State file: workstation_cloud_state.json under DELTA_DATA_DIR (updated by scripts).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import urllib.error
import urllib.request
from datetime import UTC, datetime

from fastapi import APIRouter

from tools.executor import register_handler

from .settings import WorkstationCloudSettings, data_dir, state_path

logger = logging.getLogger("deltai.extensions.workstation_cloud")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "workstation_cloud_status",
            "description": (
                "Summarize workstation cloud sync configuration: which env targets are set, "
                "state file age, and last manifest/preflight hints from workstation_cloud_state.json. "
                "Does not reveal secret values."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workstation_cloud_rclone_version",
            "description": (
                "Run `rclone version` (read-only) to verify rclone is installed. "
                "Use when checking sync tooling on this host."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def _read_state() -> dict:
    p = state_path()
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _status_payload() -> dict:
    s = WorkstationCloudSettings.load()
    st = _read_state()
    p = state_path()
    st_stat = None
    if os.path.isfile(p):
        try:
            st_stat = os.stat(p)
        except OSError:
            st_stat = None
    return {
        "data_dir": data_dir(),
        "state_file": os.path.basename(p),
        "state_present": bool(st),
        "state_updated_at": st.get("updated_at"),
        "state_file_mtime_utc": (
            datetime.fromtimestamp(st_stat.st_mtime, tz=UTC).isoformat() if st_stat else None
        ),
        "config": {
            "appwrite_endpoint_configured": bool(s.appwrite_endpoint),
            "appwrite_project_id_configured": bool(s.appwrite_project_id),
            "workstation_s3_bucket_configured": bool(s.workstation_s3_bucket),
            "rclone_remote_configured": bool(s.rclone_remote),
        },
        "last_manifest": st.get("manifest"),
        "last_preflight": st.get("preflight"),
    }


def _ping_appwrite(endpoint: str, timeout_sec: float = 5.0) -> tuple[bool, str]:
    base = endpoint.rstrip("/")
    url = f"{base}/health" if not base.endswith("/health") else base
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            code = resp.getcode()
            if 200 <= code < 300:
                return True, f"ok http {code}"
            return False, f"http {code}"
    except urllib.error.HTTPError as e:
        return False, f"http_error {e.code}"
    except urllib.error.URLError as e:
        return False, f"url_error {e.reason!r}"
    except TimeoutError:
        return False, "timeout"
    except OSError as e:
        return False, f"os_error {e}"


def _workstation_cloud_status_handler(_args: dict) -> str:
    import json as _json

    return _json.dumps(_status_payload(), indent=2)


def _workstation_cloud_rclone_version_handler(_args: dict) -> str:
    try:
        proc = subprocess.run(
            ["rclone", "version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except FileNotFoundError:
        return "rclone not found in PATH"
    except subprocess.TimeoutExpired:
        return "rclone version timed out"
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        return f"rclone exited {proc.returncode}\n{out.strip()}"
    return out.strip() or "(no output)"


def setup(app) -> None:
    register_handler("workstation_cloud_status", _workstation_cloud_status_handler)
    register_handler("workstation_cloud_rclone_version", _workstation_cloud_rclone_version_handler)

    router = APIRouter(
        prefix="/ext/workstation_cloud",
        tags=["workstation_cloud"],
    )

    @router.get("/status")
    def status():
        return _status_payload()

    @router.get("/health")
    def health():
        s = WorkstationCloudSettings.load()
        out: dict = {"appwrite": None}
        if s.appwrite_endpoint:
            ok, msg = _ping_appwrite(s.appwrite_endpoint)
            out["appwrite"] = {"reachable": ok, "detail": msg}
        else:
            out["appwrite"] = {"reachable": None, "detail": "APPWRITE_ENDPOINT not set"}
        return out

    app.include_router(router)
    logger.info("workstation_cloud: routes at /ext/workstation_cloud/")
