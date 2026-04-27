"""Unit tests for arch_update_guard path/URL guards."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.arch_update_guard.guards import (  # noqa: E402
    canonical_etc_path,
    validated_http_service_url,
)


def test_canonical_etc_path_accepts_real_under_etc():
    if not os.path.isfile("/etc/hostname"):
        pytest.skip("/etc/hostname not present")
    want = os.path.realpath("/etc/hostname")
    assert canonical_etc_path("/etc/hostname") == want


def test_canonical_etc_path_rejects_traversal(monkeypatch):
    def fake_realpath(p: str) -> str:
        if "evil" in p or "tmp" in p:
            return "/tmp/evil"
        return p

    monkeypatch.setattr("os.path.realpath", fake_realpath)
    assert canonical_etc_path("/etc/../tmp/evil") is None


def test_validated_http_service_url_ingest_loopback_only():
    d = "http://127.0.0.1:8000"
    assert validated_http_service_url("http://127.0.0.1:9999", mode="ingest", default=d) == (
        "http://127.0.0.1:9999"
    )
    assert validated_http_service_url("http://[::1]:8000/", mode="ingest", default=d) == (
        "http://[::1]:8000/"
    )
    assert validated_http_service_url("http://169.254.169.254/", mode="ingest", default=d) == d
    assert validated_http_service_url("http://example.com/", mode="ingest", default=d) == d
    assert validated_http_service_url("file:///etc/passwd", mode="ingest", default=d) == d


def test_validated_http_service_url_ollama_private_ok():
    d = "http://localhost:11434"
    assert (
        validated_http_service_url("http://10.0.0.1:11434", mode="ollama", default=d)
        == "http://10.0.0.1:11434"
    )
    assert validated_http_service_url("http://169.254.169.254/", mode="ollama", default=d) == d
