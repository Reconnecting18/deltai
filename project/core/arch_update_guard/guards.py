"""Path and URL guards for arch_update_guard (CodeQL-friendly)."""

from __future__ import annotations

import ipaddress
import os
from typing import Literal
from urllib.parse import urlsplit

_METADATA_IP = ipaddress.ip_address("169.254.169.254")


def canonical_etc_path(path: str) -> str | None:
    """Resolve *path* and return it only if the result is under ``/etc/``."""
    p = (path or "").strip()
    if not p or "\x00" in p or not os.path.isabs(p):
        return None
    try:
        rp = os.path.realpath(os.path.expanduser(p))
    except OSError:
        return None
    if rp == "/etc":
        return None
    prefix = "/etc" + os.sep
    if not rp.startswith(prefix):
        return None
    return rp


def _blocked_metadata_host(host: str) -> bool:
    h = (host or "").strip().lower()
    if h == "metadata.google.internal":
        return True
    try:
        return ipaddress.ip_address(h) == _METADATA_IP
    except ValueError:
        return False


def _host_ok_for_ingest(host: str) -> bool:
    if _blocked_metadata_host(host):
        return False
    hl = host.strip().lower()
    if hl == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host)
        return bool(ip.is_loopback)
    except ValueError:
        return False


def _host_ok_for_ollama(host: str) -> bool:
    if _blocked_metadata_host(host):
        return False
    hl = host.strip().lower()
    if hl == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_loopback or ip.is_private:
            return True
        return False
    except ValueError:
        return False


def validated_http_service_url(
    raw: str,
    *,
    mode: Literal["ingest", "ollama"],
    default: str,
) -> str:
    """
    Return *raw* if it is an http(s) URL with an allowlisted host; else *default*.

    Ingest calls stay on loopback only. Ollama may use loopback or RFC1918 hosts.
    Cloud metadata and non-TCP schemes are rejected.
    """
    d = (default or "").strip()
    if not raw or not isinstance(raw, str):
        return d
    s = raw.strip()
    if not s:
        return d
    try:
        u = urlsplit(s)
    except ValueError:
        return d
    if u.scheme not in ("http", "https"):
        return d
    host = u.hostname
    if host is None:
        return d
    ok = _host_ok_for_ingest(host) if mode == "ingest" else _host_ok_for_ollama(host)
    if not ok:
        return d
    return s
