"""
Appwrite Storage + Functions over HTTPS (httpx). Credentials from environment only.

Env (all optional until you call an API method — see status in extension):
  DELTAI_APPWRITE_ENDPOINT   e.g. https://cloud.appwrite.io/v1
  DELTAI_APPWRITE_PROJECT_ID
  DELTAI_APPWRITE_API_KEY    server / API key with storage (+ functions) scope
  DELTAI_APPWRITE_BUCKET_ID  default bucket for storage tools
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

_DEFAULT_TIMEOUT = 120.0


def _endpoint() -> str:
    return (os.getenv("DELTAI_APPWRITE_ENDPOINT") or "").strip().rstrip("/")


def _project_id() -> str:
    return (os.getenv("DELTAI_APPWRITE_PROJECT_ID") or "").strip()


def _api_key() -> str:
    return (os.getenv("DELTAI_APPWRITE_API_KEY") or "").strip()


def _bucket_id() -> str:
    return (os.getenv("DELTAI_APPWRITE_BUCKET_ID") or "").strip()


def config_status() -> dict[str, Any]:
    """Non-secret summary for appwrite_status tool."""
    ep = _endpoint()
    key = _api_key()
    return {
        "endpoint_configured": bool(ep),
        "endpoint_host": ep.split("//")[-1].split("/")[0] if ep else "",
        "project_configured": bool(_project_id()),
        "api_key_configured": bool(key),
        "bucket_configured": bool(_bucket_id()),
    }


def _require_config(*, need_bucket: bool) -> None:
    if not _endpoint() or not _project_id() or not _api_key():
        raise ValueError(
            "Set DELTAI_APPWRITE_ENDPOINT, DELTAI_APPWRITE_PROJECT_ID, and DELTAI_APPWRITE_API_KEY"
        )
    if need_bucket and not _bucket_id():
        raise ValueError("Set DELTAI_APPWRITE_BUCKET_ID for storage operations")


def _headers() -> dict[str, str]:
    return {
        "X-Appwrite-Project": _project_id(),
        "X-Appwrite-Key": _api_key(),
    }


def storage_list(*, bucket_id: str | None = None, limit: int = 25, offset: int = 0) -> dict[str, Any]:
    _require_config(need_bucket=False)
    bid = (bucket_id or _bucket_id()).strip()
    if not bid:
        raise ValueError("bucket_id required (or set DELTAI_APPWRITE_BUCKET_ID)")
    url = f"{_endpoint()}/storage/buckets/{bid}/files"
    params: dict[str, Any] = {"limit": max(1, min(100, int(limit))), "offset": max(0, int(offset))}
    with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
        r = client.get(url, headers=_headers(), params=params)
    r.raise_for_status()
    return r.json()


def storage_upload(
    *,
    local_path: str,
    bucket_id: str | None = None,
    file_id: str | None = None,
) -> dict[str, Any]:
    _require_config(need_bucket=False)
    bid = (bucket_id or _bucket_id()).strip()
    if not bid:
        raise ValueError("bucket_id required (or set DELTAI_APPWRITE_BUCKET_ID)")
    fid = (file_id or "unique()").strip() or "unique()"
    url = f"{_endpoint()}/storage/buckets/{bid}/files"
    filename = os.path.basename(local_path) or "upload.bin"
    with open(local_path, "rb") as f:
        files = {"file": (filename, f, "application/octet-stream")}
        data = {"fileId": fid}
        with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
            r = client.post(url, headers=_headers(), data=data, files=files)
    if r.status_code >= 400:
        try:
            detail = r.json()
        except json.JSONDecodeError:
            detail = {"raw": (r.text or "")[:2000]}
        raise ValueError(f"appwrite upload failed: {r.status_code} {detail}")
    return r.json()


def storage_download(
    *,
    file_id: str,
    local_path: str,
    bucket_id: str | None = None,
) -> dict[str, Any]:
    _require_config(need_bucket=False)
    bid = (bucket_id or _bucket_id()).strip()
    if not bid:
        raise ValueError("bucket_id required (or set DELTAI_APPWRITE_BUCKET_ID)")
    fid = (file_id or "").strip()
    if not fid:
        raise ValueError("file_id required")
    url = f"{_endpoint()}/storage/buckets/{bid}/files/{fid}/download"
    with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
        r = client.get(url, headers=_headers())
    r.raise_for_status()
    parent = os.path.dirname(os.path.abspath(local_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(local_path, "wb") as out:
        out.write(r.content)
    return {"ok": True, "bytes": len(r.content), "path": local_path}


def storage_delete(*, file_id: str, bucket_id: str | None = None) -> dict[str, Any]:
    _require_config(need_bucket=False)
    bid = (bucket_id or _bucket_id()).strip()
    if not bid:
        raise ValueError("bucket_id required (or set DELTAI_APPWRITE_BUCKET_ID)")
    fid = (file_id or "").strip()
    if not fid:
        raise ValueError("file_id required")
    url = f"{_endpoint()}/storage/buckets/{bid}/files/{fid}"
    with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
        r = client.delete(url, headers=_headers())
    if r.status_code >= 400:
        try:
            detail = r.json()
        except json.JSONDecodeError:
            detail = {"raw": (r.text or "")[:2000]}
        raise ValueError(f"appwrite delete failed: {r.status_code} {detail}")
    return r.json() if r.content else {"ok": True}


def function_execute(
    *,
    function_id: str,
    body: str = "{}",
    async_execution: bool = False,
) -> dict[str, Any]:
    _require_config(need_bucket=False)
    fid = (function_id or "").strip()
    if not fid:
        raise ValueError("function_id required")
    url = f"{_endpoint()}/functions/{fid}/executions"
    payload = {"body": body, "async": bool(async_execution)}
    with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
        r = client.post(url, headers={**_headers(), "Content-Type": "application/json"}, json=payload)
    if r.status_code >= 400:
        try:
            detail = r.json()
        except json.JSONDecodeError:
            detail = {"raw": (r.text or "")[:2000]}
        raise ValueError(f"appwrite execution failed: {r.status_code} {detail}")
    return r.json()
