"""
arch_update_guard news/wiki ingest helpers.

Fetches Arch Linux RSS + ArchWiki context and pushes a digest into deltai via
POST /ingest so RAG can include update-risk context in extension decisions.
"""

from __future__ import annotations

import logging
import threading
import time
import xml.etree.ElementTree as ET
from typing import Any

import httpx
import safe_errors

logger = logging.getLogger("deltai.extensions.arch_update_guard.news")

ARCH_NEWS_RSS = "https://archlinux.org/feeds/news/"
WIKI_API = "https://wiki.archlinux.org/api.php"
DEFAULT_BASE = "http://127.0.0.1:8000"

# Simple in-process rate limit for RSS/wiki fetches (seconds)
_MIN_REFRESH_INTERVAL = 45.0
_last_refresh_ts: float = 0.0
_refresh_lock = threading.Lock()


def _base_url() -> str:
    import os

    return os.environ.get("DELTAI_BASE_URL", DEFAULT_BASE).rstrip("/")


def _ingest_sync(source: str, context: str, *, ttl: int, tags: list[str]) -> dict[str, Any]:
    """POST one ingest item using synchronous httpx (tool handlers and threads)."""
    url = f"{_base_url()}/ingest"
    try:
        with httpx.Client(timeout=120.0) as client:
            r = client.post(
                url,
                json={"source": source, "context": context, "ttl": ttl, "tags": tags},
            )
            try:
                body = r.json()
            except Exception:
                body = {"raw": r.text[:500]}
            return {"http_status": r.status_code, "body": body}
    except httpx.RequestError as exc:
        safe_errors.log_exception(logger, "ingest request failed", exc)
        return {
            "http_status": None,
            "body": {
                "status": "error",
                "reason": safe_errors.public_error_detail(exc, generic="ingest request failed"),
            },
        }


async def _ingest_async(source: str, context: str, *, ttl: int, tags: list[str]) -> dict[str, Any]:
    url = f"{_base_url()}/ingest"
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                url,
                json={"source": source, "context": context, "ttl": ttl, "tags": tags},
            )
            try:
                body = r.json()
            except Exception:
                body = {"raw": r.text[:500]}
            return {"http_status": r.status_code, "body": body}
    except httpx.RequestError as exc:
        safe_errors.log_exception(logger, "ingest request failed", exc)
        return {
            "http_status": None,
            "body": {
                "status": "error",
                "reason": safe_errors.public_error_detail(exc, generic="ingest request failed"),
            },
        }


def fetch_arch_news_rss(timeout: float = 30.0) -> tuple[list[dict[str, str]], str | None]:
    """Download and parse the official Arch news RSS feed."""
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            r = client.get(ARCH_NEWS_RSS)
            r.raise_for_status()
            text = r.text
    except Exception as exc:
        safe_errors.log_exception(logger, "arch news RSS fetch failed", exc)
        return [], safe_errors.public_error_detail(exc, generic="rss fetch failed")

    items: list[dict[str, str]] = []
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        safe_errors.log_exception(logger, "RSS parse error", exc)
        return [], "RSS parse error"

    # RSS 2.0: channel/item
    channel = root.find("channel")
    if channel is not None:
        for el in channel.findall("item"):
            title = (el.findtext("title") or "").strip()
            link = (el.findtext("link") or "").strip()
            desc = (el.findtext("description") or "").strip()
            pub = (el.findtext("pubDate") or "").strip()
            if title:
                items.append({"title": title, "link": link, "description": desc, "pubDate": pub})
        return items, None

    return [], "No RSS channel/items found (unexpected feed shape)"


def fetch_wiki_snippet(query: str, timeout: float = 25.0) -> tuple[str | None, str | None]:
    """Return plain-text extract for the best wiki search hit, or (None, error)."""
    q = query.strip()
    if not q:
        return None, "empty wiki_query"

    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": q,
        "srlimit": "1",
    }
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            r = client.get(WIKI_API, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception as exc:
        safe_errors.log_exception(logger, "wiki search failed", exc)
        return None, safe_errors.public_error_detail(exc, generic="wiki search failed")

    hits = data.get("query", {}).get("search", [])
    if not hits:
        return None, "no wiki hits"

    title = hits[0].get("title")
    if not title:
        return None, "empty title from wiki"

    ep = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "exintro": "1",
        "explaintext": "1",
        "titles": title,
    }
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            r2 = client.get(WIKI_API, params=ep)
            r2.raise_for_status()
            data2 = r2.json()
    except Exception as exc:
        safe_errors.log_exception(logger, "wiki extract fetch failed", exc)
        return None, safe_errors.public_error_detail(exc, generic="wiki extract fetch failed")

    pages = data2.get("query", {}).get("pages", {})
    for _pid, page in pages.items():
        extract = (page.get("extract") or "").strip()
        if extract:
            return f"Title: {title}\n\n{extract}", None
    return None, "no extract returned"


def build_news_digest(items: list[dict[str, str]], max_items: int = 12) -> str:
    lines: list[str] = ["Arch Linux — official news feed (for operator / RAG context)", ""]
    for it in items[:max_items]:
        lines.append(f"## {it['title']}")
        if it.get("pubDate"):
            lines.append(f"Date: {it['pubDate']}")
        if it.get("link"):
            lines.append(f"Link: {it['link']}")
        desc = it.get("description") or ""
        if desc:
            # strip simple HTML tags if present
            plain = desc.replace("<p>", "\n").replace("</p>", "\n")
            plain = plain.replace("<li>", "- ").replace("</li>", "\n")
            lines.append(plain.strip())
        lines.append("")
    return "\n".join(lines).strip()


def refresh_news_digest_to_rag(
    *,
    wiki_query: str = "",
    force: bool = False,
    news_ttl_sec: int = 86400 * 14,
) -> dict[str, Any]:
    """
    Fetch RSS (+ optional wiki), POST combined digest to /ingest.
    Respects a short in-process rate limit unless force=True.
    """
    global _last_refresh_ts
    now = time.monotonic()
    with _refresh_lock:
        if not force and (now - _last_refresh_ts) < _MIN_REFRESH_INTERVAL:
            return {
                "status": "skipped",
                "reason": (
                    f"rate_limited (min interval {_MIN_REFRESH_INTERVAL}s; use force=true to override)"
                ),
            }
        _last_refresh_ts = now

    items, err = fetch_arch_news_rss()
    if err:
        return {"status": "error", "reason": f"rss: {err}"}

    digest = build_news_digest(items)
    wiki_part = ""
    werr: str | None = None
    if wiki_query.strip():
        snippet, werr = fetch_wiki_snippet(wiki_query.strip())
        if snippet:
            wiki_part = "\n\n---\n\nArchWiki snippet:\n\n" + snippet

    combined = digest + wiki_part
    ing = _ingest_sync(
        "arch_news",
        combined,
        ttl=news_ttl_sec,
        tags=["arch", "news", "ingest"],
    )

    out: dict[str, Any] = {
        "status": "ok",
        "items_fetched": len(items),
        "ingest": ing,
    }
    if wiki_query.strip() and werr:
        out["wiki_warning"] = werr
    return out


async def refresh_news_digest_to_rag_async(
    *,
    wiki_query: str = "",
    force: bool = False,
    news_ttl_sec: int = 86400 * 14,
) -> dict[str, Any]:
    """Async variant for FastAPI routes (same logic as refresh_news_digest_to_rag)."""
    global _last_refresh_ts
    now = time.monotonic()
    with _refresh_lock:
        if not force and (now - _last_refresh_ts) < _MIN_REFRESH_INTERVAL:
            return {
                "status": "skipped",
                "reason": (
                    f"rate_limited (min interval {_MIN_REFRESH_INTERVAL}s; use force=true to override)"
                ),
            }
        _last_refresh_ts = now

    # Run blocking fetch in thread to not block event loop
    import asyncio

    items, err = await asyncio.to_thread(fetch_arch_news_rss)
    if err:
        return {"status": "error", "reason": f"rss: {err}"}

    digest = build_news_digest(items)
    wiki_part = ""
    werr: str | None = None
    if wiki_query.strip():
        snippet, werr = await asyncio.to_thread(fetch_wiki_snippet, wiki_query.strip())
        if snippet:
            wiki_part = "\n\n---\n\nArchWiki snippet:\n\n" + snippet

    combined = digest + wiki_part
    ing = await _ingest_async(
        "arch_news",
        combined,
        ttl=news_ttl_sec,
        tags=["arch", "news", "ingest"],
    )

    out: dict[str, Any] = {
        "status": "ok",
        "items_fetched": len(items),
        "ingest": ing,
    }
    if wiki_query.strip() and werr:
        out["wiki_warning"] = werr
    return out
