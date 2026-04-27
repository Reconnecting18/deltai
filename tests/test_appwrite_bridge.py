"""Tests for Appwrite bridge extension (httpx client + tool handlers)."""

from __future__ import annotations

import json
import os
from unittest import mock

import httpx

# Extension lives under project/extensions; pytest pythonpath includes project.
from extensions.appwrite_bridge import api as aw_api


def test_config_status_empty():
    with mock.patch.dict(os.environ, {}, clear=True):
        for k in (
            "DELTAI_APPWRITE_ENDPOINT",
            "DELTAI_APPWRITE_PROJECT_ID",
            "DELTAI_APPWRITE_API_KEY",
            "DELTAI_APPWRITE_BUCKET_ID",
        ):
            os.environ.pop(k, None)
        st = aw_api.config_status()
    assert st["endpoint_configured"] is False
    assert st["api_key_configured"] is False


def test_storage_list_mocked():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/storage/buckets/mybucket/files" in str(request.url)
        return httpx.Response(200, json={"total": 0, "files": []})

    transport = httpx.MockTransport(handler)
    with mock.patch.dict(
        os.environ,
        {
            "DELTAI_APPWRITE_ENDPOINT": "https://example.test/v1",
            "DELTAI_APPWRITE_PROJECT_ID": "p1",
            "DELTAI_APPWRITE_API_KEY": "k1",
            "DELTAI_APPWRITE_BUCKET_ID": "mybucket",
        },
    ):
        real_client = httpx.Client

        def _client(**kw):
            return real_client(transport=transport, timeout=kw.get("timeout", 120.0))

        with mock.patch("extensions.appwrite_bridge.api.httpx.Client", _client):
            out = aw_api.storage_list(bucket_id="mybucket")
    assert out["files"] == []


def test_function_execute_payload_uses_body_field():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(201, json={"$id": "e1", "status": "completed"})

    transport = httpx.MockTransport(handler)
    with mock.patch.dict(
        os.environ,
        {
            "DELTAI_APPWRITE_ENDPOINT": "https://example.test/v1",
            "DELTAI_APPWRITE_PROJECT_ID": "p1",
            "DELTAI_APPWRITE_API_KEY": "k1",
        },
    ):
        real_client = httpx.Client

        def _client(**kw):
            return real_client(transport=transport, timeout=kw.get("timeout", 120.0))

        with mock.patch("extensions.appwrite_bridge.api.httpx.Client", _client):
            aw_api.function_execute(function_id="fn1", body='{"x":1}', async_execution=True)
    assert captured["body"] == {"body": '{"x":1}', "async": True}


def test_validate_appwrite_file_id():
    from extensions.appwrite_bridge import _validate_appwrite_file_id

    assert _validate_appwrite_file_id("unique()") == "unique()"
    assert _validate_appwrite_file_id("my-file_1") == "my-file_1"
    assert _validate_appwrite_file_id("bad id") is None
