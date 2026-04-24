"""
Local server inventory and bounded SSH helpers for the server_network extension.

State lives under DELTA_DATA_DIR (default ~/.local/share/deltai) as JSON.
No root, no arbitrary hosts: remote commands only run against registered servers.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import subprocess
import uuid
from datetime import UTC, datetime
from typing import Any

import path_guard

logger = logging.getLogger("deltai.extensions.server_network")

_HOST_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9.-]*[a-zA-Z0-9]$|^[a-zA-Z0-9]$")
_IPV4_RE = re.compile(r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$")
_USER_RE = re.compile(r"^[a-z_][a-z0-9_-]{0,31}$", re.IGNORECASE)
_LABEL_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")


def _home_real() -> str:
    return os.path.realpath(os.path.expanduser("~"))


def _data_dir() -> str:
    """
    Registry root must resolve under the real home directory (path-injection safe).
    """
    home = _home_real()
    raw = (os.getenv("DELTA_DATA_DIR") or "").strip()
    if not raw:
        return path_guard.realpath_under(home, os.path.join(os.path.expanduser("~"), ".local", "share", "deltai"))
    return path_guard.realpath_under(home, os.path.expanduser(raw))


def _resolve_identity_file(path: str) -> str:
    """Private key path must be a regular file under the real home directory."""
    home = _home_real()
    resolved = path_guard.realpath_under(home, os.path.expanduser(path.strip()))
    if not os.path.isfile(resolved):
        raise ValueError("identity_file must exist and be a regular file")
    return resolved


def registry_path() -> str:
    return os.path.join(_data_dir(), "local_server_network.json")


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_host(host: str) -> str:
    raw = (host or "").strip()
    h = raw.strip("[]")
    if not h or len(h) > 253:
        raise ValueError("invalid host")
    m4 = _IPV4_RE.match(h)
    if m4:
        octets = [int(m4.group(i)) for i in range(1, 5)]
        if any(o > 255 for o in octets):
            raise ValueError("invalid IPv4 address")
        return h
    if ":" in h:
        try:
            ipaddress.IPv6Address(h)
        except ValueError as exc:
            raise ValueError("invalid IPv6 address") from exc
        return raw if raw.startswith("[") and raw.endswith("]") else h
    if not _HOST_RE.match(h):
        raise ValueError("host must be a hostname, IPv4/IPv6, or single-label name (letters, digits, dot, hyphen)")
    return h


def _validate_user(user: str) -> str:
    u = (user or "").strip()
    if not u or not _USER_RE.match(u):
        raise ValueError("invalid ssh user")
    return u


def _validate_label(label: str | None) -> str | None:
    if label is None:
        return None
    s = label.strip()
    if not s:
        return None
    if not _LABEL_RE.match(s):
        raise ValueError("invalid label")
    return s


def _load_raw() -> dict[str, Any]:
    path = registry_path()
    if not os.path.isfile(path):
        return {"schema_version": 1, "servers": []}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("server_network: could not read %s: %s", path, exc)
        return {"schema_version": 1, "servers": []}
    if not isinstance(data, dict):
        return {"schema_version": 1, "servers": []}
    data.setdefault("schema_version", 1)
    data.setdefault("servers", [])
    if not isinstance(data["servers"], list):
        data["servers"] = []
    return data


def _save_raw(data: dict[str, Any]) -> None:
    path = registry_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def list_servers() -> list[dict[str, Any]]:
    data = _load_raw()
    return list(data.get("servers", []))


def get_server(server_id: str) -> dict[str, Any] | None:
    sid = (server_id or "").strip()
    for s in list_servers():
        if s.get("id") == sid:
            return dict(s)
    return None


def add_server(
    host: str,
    user: str,
    port: int = 22,
    label: str | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
    identity_file: str | None = None,
) -> dict[str, Any]:
    h = _validate_host(host)
    u = _validate_user(user)
    if port != int(port) or port < 1 or port > 65535:
        raise ValueError("port must be 1-65535")
    port = int(port)
    lb = _validate_label(label)
    raw_tags: list[str] = []
    if tags:
        for t in tags:
            ts = str(t).strip()
            if ts and ts not in raw_tags and len(ts) <= 64:
                raw_tags.append(ts)
    notes_s = (notes or "").strip()[:4000] if notes else ""
    id_path = None
    if identity_file:
        id_path = _resolve_identity_file(str(identity_file))

    data = _load_raw()
    servers: list[dict[str, Any]] = data["servers"]
    for s in servers:
        if s.get("host") == h and int(s.get("port", 22)) == port and s.get("user") == u:
            raise ValueError("a server with this host, port, and user already exists")

    rec = {
        "id": str(uuid.uuid4()),
        "label": lb or h,
        "host": h,
        "user": u,
        "port": port,
        "tags": raw_tags,
        "notes": notes_s,
        "identity_file": id_path,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
    }
    servers.append(rec)
    _save_raw(data)
    return dict(rec)


def update_server(
    server_id: str,
    label: str | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
    identity_file: str | None = None,
    clear_identity_file: bool = False,
) -> dict[str, Any]:
    sid = (server_id or "").strip()
    data = _load_raw()
    servers: list[dict[str, Any]] = data["servers"]
    for _i, s in enumerate(servers):
        if s.get("id") != sid:
            continue
        if label is not None:
            lb = _validate_label(label)
            if lb:
                s["label"] = lb
        if tags is not None:
            raw_tags: list[str] = []
            for t in tags:
                ts = str(t).strip()
                if ts and ts not in raw_tags and len(ts) <= 64:
                    raw_tags.append(ts)
            s["tags"] = raw_tags
        if notes is not None:
            s["notes"] = str(notes).strip()[:4000]
        if clear_identity_file:
            s["identity_file"] = None
        elif identity_file is not None:
            s["identity_file"] = _resolve_identity_file(str(identity_file))
        s["updated_at"] = _utc_now()
        _save_raw(data)
        return dict(s)
    raise ValueError("server not found")


def remove_server(server_id: str) -> bool:
    sid = (server_id or "").strip()
    data = _load_raw()
    servers: list[dict[str, Any]] = data["servers"]
    new_list = [s for s in servers if s.get("id") != sid]
    if len(new_list) == len(servers):
        return False
    data["servers"] = new_list
    _save_raw(data)
    return True


def _ssh_base(rec: dict[str, Any]) -> list[str]:
    # Default OpenSSH host key policy (verify known_hosts); do not use accept-new/no
    # so static analysis and CI do not flag weakened host-key validation.
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=12",
    ]
    port = int(rec.get("port") or 22)
    if port != 22:
        cmd.extend(["-p", str(port)])
    idf = rec.get("identity_file")
    if idf:
        cmd.extend(["-i", str(idf)])
    user = str(rec.get("user") or "")
    host = str(rec.get("host") or "")
    # OpenSSH requires IPv6 literals in brackets when not already bracketed.
    if ":" in host and "[" not in host:
        target = f"{user}@[{host}]"
    else:
        target = f"{user}@{host}"
    cmd.append(target)
    return cmd


def probe_server(server_id: str, timeout_sec: int = 8) -> dict[str, Any]:
    rec = get_server(server_id)
    if not rec:
        raise ValueError("server not found")
    cmd = _ssh_base(rec) + ["true"]
    try:
        proc = subprocess.run(
            cmd,
            shell=False,
            capture_output=True,
            text=True,
            timeout=max(3, min(60, int(timeout_sec))),
        )
    except subprocess.TimeoutExpired:
        return {"server_id": server_id, "reachable": False, "error": "timeout"}
    except OSError as exc:
        return {"server_id": server_id, "reachable": False, "error": str(exc)}
    ok = proc.returncode == 0
    err = (proc.stderr or "").strip()[:2000] if not ok else ""
    return {"server_id": server_id, "reachable": ok, "returncode": proc.returncode, "stderr": err}


def run_remote_command(server_id: str, command: str, timeout_sec: int = 120) -> dict[str, Any]:
    rec = get_server(server_id)
    if not rec:
        raise ValueError("server not found")
    cmd_line = (command or "").strip()
    if not cmd_line:
        raise ValueError("command must be non-empty")
    if len(cmd_line) > 16000:
        raise ValueError("command too long")
    cmd = _ssh_base(rec) + [cmd_line]
    try:
        proc = subprocess.run(
            cmd,
            shell=False,
            capture_output=True,
            text=True,
            timeout=max(5, min(600, int(timeout_sec))),
        )
    except subprocess.TimeoutExpired:
        return {"server_id": server_id, "timeout": True}
    return {
        "server_id": server_id,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "")[-24000:],
        "stderr": (proc.stderr or "")[-12000:],
    }


def run_remote_script(server_id: str, script: str, timeout_sec: int = 300) -> dict[str, Any]:
    rec = get_server(server_id)
    if not rec:
        raise ValueError("server not found")
    body = script or ""
    if not body.strip():
        raise ValueError("script must be non-empty")
    if len(body) > 64000:
        raise ValueError("script too long")
    cmd = _ssh_base(rec) + ["bash", "-s"]
    try:
        proc = subprocess.run(
            cmd,
            shell=False,
            input=body,
            capture_output=True,
            text=True,
            timeout=max(10, min(900, int(timeout_sec))),
        )
    except subprocess.TimeoutExpired:
        return {"server_id": server_id, "timeout": True}
    return {
        "server_id": server_id,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "")[-24000:],
        "stderr": (proc.stderr or "")[-12000:],
    }
