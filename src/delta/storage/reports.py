"""User-visible JSON reports for AI orchestration and chat (on-disk transparency)."""

from __future__ import annotations

import json
import logging
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger("delta.reports")

REPORT_SCHEMA_VERSION = 1

ReportCategory = Literal["orchestrator", "chat"]


def _env_flag_enabled(var_name: str, default: bool = True) -> bool:
    raw = os.getenv(var_name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def resolve_reports_dir_from_env() -> Path:
    """Match load_settings defaults so project/ can write without importing Settings."""
    xdg_data = Path(os.getenv("XDG_DATA_HOME", str(Path.home() / ".local/share")))
    data_dir = Path(os.getenv("DELTA_DATA_DIR", str(xdg_data / "deltai"))).expanduser()
    override = os.getenv("DELTA_REPORTS_DIR")
    if override:
        return Path(override).expanduser()
    return (data_dir / "ai_reports").expanduser()


def reports_enabled_from_env() -> bool:
    return _env_flag_enabled("DELTA_AI_REPORTS", default=True)


def ensure_reports_layout(reports_dir: Path) -> None:
    for name in ("orchestrator", "chat", "errors"):
        (reports_dir / name).mkdir(parents=True, exist_ok=True)


def _report_filename() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = secrets.token_hex(4)
    return f"{ts}_{suffix}.json"


def write_ai_report(
    *,
    reports_dir: Path,
    enabled: bool,
    category: ReportCategory,
    status: str,
    fields: dict[str, Any],
) -> Path | None:
    """Write one JSON report; never raises. Returns path written or None."""
    if not enabled:
        return None
    try:
        ensure_reports_layout(reports_dir)
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        subdir = reports_dir / category / day
        subdir.mkdir(parents=True, exist_ok=True)
        fname = _report_filename()
        doc: dict[str, Any] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "written_at": datetime.now(timezone.utc).isoformat(),
            "source": category,
            "status": status,
            **fields,
        }
        primary = subdir / fname
        _atomic_write_json(primary, doc)
        if status == "error":
            err_subdir = reports_dir / "errors" / day
            err_subdir.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(err_subdir / fname, doc)
        return primary
    except OSError as exc:
        logger.warning("AI report write failed [%s]: %s", type(exc).__name__, exc)
        return None
    except (TypeError, ValueError) as exc:
        logger.warning("AI report serialization failed [%s]: %s", type(exc).__name__, exc)
        return None


def _atomic_write_json(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(doc, indent=2, ensure_ascii=False, default=str)
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def write_chat_turn_report(
    *,
    user_message: str,
    assistant_response: str,
    chat_metadata: dict[str, Any] | None = None,
    status: str = "ok",
    user_visible_error: str | None = None,
    internal_detail: str | None = None,
) -> None:
    """Write a chat report using DELTA_DATA_DIR / DELTA_REPORTS_DIR / DELTA_AI_REPORTS."""
    reports_dir = resolve_reports_dir_from_env()
    enabled = reports_enabled_from_env()
    fields: dict[str, Any] = {
        "query": user_message,
        "output": assistant_response,
        "chat_metadata": chat_metadata or {},
    }
    if user_visible_error:
        fields["error"] = {"user_message": user_visible_error}
        if internal_detail:
            fields["error"]["detail"] = internal_detail
    write_ai_report(
        reports_dir=reports_dir,
        enabled=enabled,
        category="chat",
        status=status,
        fields=fields,
    )
