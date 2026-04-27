"""
Environment-driven settings for workstation_cloud extension.

Secrets (e.g. APPWRITE_API_KEY) are read only when needed for future hooks;
they must never be logged or returned from HTTP handlers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import path_guard


def data_dir() -> str:
    """DELTA_DATA_DIR under the real home directory (path-injection safe)."""
    home = os.path.realpath(os.path.expanduser("~"))
    raw = (os.getenv("DELTA_DATA_DIR") or "").strip()
    if not raw:
        return path_guard.realpath_under(
            home,
            os.path.join(os.path.expanduser("~"), ".local", "share", "deltai"),
        )
    return path_guard.realpath_under(home, os.path.expanduser(raw))


def state_path() -> str:
    return os.path.join(data_dir(), "workstation_cloud_state.json")


@dataclass(frozen=True)
class WorkstationCloudSettings:
    appwrite_endpoint: str | None
    appwrite_project_id: str | None
    workstation_s3_bucket: str | None
    rclone_remote: str | None

    @staticmethod
    def load() -> WorkstationCloudSettings:
        ep = (os.getenv("APPWRITE_ENDPOINT") or "").strip() or None
        pid = (os.getenv("APPWRITE_PROJECT_ID") or "").strip() or None
        bucket = (os.getenv("WORKSTATION_S3_BUCKET") or "").strip() or None
        remote = (os.getenv("WORKSTATION_RCLONE_REMOTE") or "").strip() or None
        if remote and not _safe_remote_name(remote):
            remote = None
        return WorkstationCloudSettings(
            appwrite_endpoint=ep,
            appwrite_project_id=pid,
            workstation_s3_bucket=bucket,
            rclone_remote=remote,
        )


def _safe_remote_name(name: str) -> bool:
    if not name or len(name) > 64:
        return False
    return all(c.isalnum() or c in ("_", "-") for c in name)
