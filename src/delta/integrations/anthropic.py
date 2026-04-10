"""Anthropic cloud fallback adapter.

This module should wrap reusable behavior from deltai's anthropic client,
including model fallback and structured error handling.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


@dataclass
class AnthropicClient:
    """Minimal async Anthropic client for fallback inference."""

    api_key: str
    model: str

    async def complete(self, message: str, max_tokens: int = 1024) -> str:
        """Run a simple cloud completion request."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": message}],
        }

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(ANTHROPIC_API_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        content = data.get("content", [])
        text_blocks = [b.get("text", "") for b in content if b.get("type") == "text"]
        return "".join(text_blocks).strip()
