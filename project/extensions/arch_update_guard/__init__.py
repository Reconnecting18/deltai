"""
Arch update guard — extension for Arch Linux news/wiki context and pacman evidence.

See README.md for scope, no-root policy, and optional ALPM hook template.
"""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel

from .news_wiki import refresh_news_digest_to_rag, refresh_news_digest_to_rag_async
from .pacman_audit import get_pending_updates

logger = logging.getLogger("deltai.extensions.arch_update_guard")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "arch_pending_updates_report",
            "description": (
                "Arch Linux: list pending package upgrades using checkupdates or pacman -Qu, "
                "with optional reverse-dependency hints (pactree). Returns JSON evidence — not a "
                "prediction that the system will or will not break."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "include_reverse_deps": {
                        "type": "boolean",
                        "description": (
                            "If true, run pactree -ru for the first few pending packages (slower)."
                        ),
                        "default": False,
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "arch_refresh_news_digest",
            "description": (
                "Fetch official Arch Linux news RSS (optional ArchWiki snippet), "
                "POST digest to deltai RAG via /ingest (source arch_news). "
                "Use before discussing upgrades."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "wiki_query": {
                        "type": "string",
                        "description": "Optional ArchWiki search phrase (e.g. pacnew, mkinitcpio).",
                        "default": "",
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Bypass short in-process rate limit between refreshes.",
                        "default": False,
                    },
                },
                "required": [],
            },
        },
    },
]


class RefreshNewsBody(BaseModel):
    wiki_query: str = ""
    force: bool = False


def _arch_pending_updates_report_handler(include_reverse_deps: bool = False) -> str:
    data = get_pending_updates(
        include_reverse_deps=include_reverse_deps,
        reverse_deps_limit=5,
    )
    return json.dumps(data, indent=2)


def _arch_refresh_news_digest_handler(wiki_query: str = "", force: bool = False) -> str:
    result = refresh_news_digest_to_rag(wiki_query=wiki_query, force=force)
    return json.dumps(result, indent=2, default=str)


def setup(app) -> None:
    from fastapi import APIRouter, Request
    from tools.executor import register_handler

    register_handler("arch_pending_updates_report", _arch_pending_updates_report_handler)
    register_handler("arch_refresh_news_digest", _arch_refresh_news_digest_handler)

    router = APIRouter(prefix="/ext/arch_update_guard", tags=["arch_update_guard"])

    @router.get("/pending")
    def pending(
        include_reverse_deps: bool = False,
        reverse_deps_limit: int = 5,
    ):
        return get_pending_updates(
            include_reverse_deps=include_reverse_deps,
            reverse_deps_limit=reverse_deps_limit,
        )

    @router.post("/refresh-news")
    async def refresh_news(request: Request):
        try:
            raw = await request.json()
            if not isinstance(raw, dict):
                raw = {}
        except Exception:
            raw = {}
        body = RefreshNewsBody.model_validate(raw)
        return await refresh_news_digest_to_rag_async(
            wiki_query=body.wiki_query,
            force=body.force,
        )

    @router.get("/health")
    def health():
        return {"status": "ok", "extension": "arch_update_guard"}

    app.include_router(router)
    logger.info("arch_update_guard: routes at /ext/arch_update_guard/")


def shutdown() -> None:
    logger.info("arch_update_guard: shutdown")
