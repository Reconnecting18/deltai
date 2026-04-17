"""Unit tests for project.url_safety (SSRF gate for fetch_url)."""

import pytest

from url_safety import validate_http_url_for_fetch


def test_rejects_loopback_literal() -> None:
    with pytest.raises(ValueError):
        validate_http_url_for_fetch("http://127.0.0.1/")


def test_rejects_metadata_ip() -> None:
    with pytest.raises(ValueError):
        validate_http_url_for_fetch("http://169.254.169.254/latest/meta-data/")


def test_rejects_non_http_scheme() -> None:
    with pytest.raises(ValueError):
        validate_http_url_for_fetch("file:///etc/passwd")


def test_accepts_public_https_url() -> None:
    validate_http_url_for_fetch("https://example.com/path")
