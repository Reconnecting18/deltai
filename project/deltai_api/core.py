import asyncio
import json
import logging
import os
import re
import time as _time
from collections import deque
from contextlib import asynccontextmanager

import httpx
import psutil
import safe_errors
from anthropic_client import split_workload_planner_outline
from anthropic_client import stream_chat as anthropic_stream
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, WebSocket
from path_guard import realpath_under
from persistence import (
    _serialize_embedding,
    find_similar_traces,
    get_unresolved_gaps,
    init_db,
    load_history,
    prune_old_traces,
    resolve_knowledge_gap,
    save_history_pair,
    save_knowledge_gap,
    save_quality_score,
    save_reasoning_trace,
    save_routing_feedback,
    trim_history,
)
from persistence import clear_history as db_clear_history
from prompts import (
    build_local_system_prompt,
    build_react_system_prompt,
    protocol_antifabrication_reminder,
)
from pydantic import BaseModel
from router import (
    _pick_local_model,
    check_model_exists,
    check_model_health,
    get_backup_model,
    get_budget_status,
    get_gpu_utilization,
    get_vram_free_mb,
    init_budget_from_db,
    is_cloud_available,
    is_cloud_available_sync,
    is_sim_running,
    record_cloud_usage,
    route,
)
from tools.definitions import TOOLS, filter_tools
from tools.executor import execute_tool

load_dotenv()
psutil.cpu_percent(interval=0.1)

logger = logging.getLogger("deltai")

try:
    import pynvml

    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

# ── RAG IMPORTS ─────────────────────────────────────────────────────────
RAG_AVAILABLE = False
try:
    from memory import (
        KNOWLEDGE_PATH,
        cleanup_expired,
        compact_warm_to_cold,
        get_cold_stats,
        get_file_details,
        get_memory_stats,
        ingest_all,
        ingest_context,
        ingest_context_batch,
        query_knowledge,
        remove_file,
    )
    from watcher import start_watcher, stop_watcher, watcher_running

    RAG_AVAILABLE = True
except ImportError as e:
    logger.warning("RAG system unavailable [%s]", type(e).__name__)

# ── TRAINING IMPORTS ───────────────────────────────────────────────────
TRAINING_AVAILABLE = False
try:
    from training import (
        add_example,
        auto_capture,
        create_dataset,
        delete_dataset,
        export_dataset,
        get_dataset,
        get_training_status,
        list_datasets,
        remove_example,
        start_training,
        stop_training,
    )

    TRAINING_AVAILABLE = True
except ImportError as e:
    logger.warning("Training system unavailable [%s]", type(e).__name__)


# ── EXTENSIONS IMPORTS ─────────────────────────────────────────────────
EXTENSIONS_AVAILABLE = False
try:
    from extensions import get_extension_tools, load_extensions, shutdown_extensions
    from tools.definitions import _merge_extension_tools

    EXTENSIONS_AVAILABLE = True
except ImportError as e:
    logger.warning("Extensions system unavailable [%s]", type(e).__name__)

# ── ARCH UPDATE GUARD (optional extension) ────────────────────────────
# See project/extensions/arch_update_guard/ — present on personal; gitignored on main.
arch_guard_scheduler_loop = None
try:
    from extensions.arch_update_guard.scheduler import (
        scheduler_loop as arch_guard_scheduler_loop,
    )
except ImportError:
    pass

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DELTAI_MODEL = os.getenv("DELTAI_MODEL", "deltai")
BACKUP_MAX_RETRIES = int(os.getenv("BACKUP_MAX_RETRIES", "2"))
TELEMETRY_API_URL = os.getenv("TELEMETRY_API_URL", "").strip()
# Optional: when set, POST /ingest, /ingest/batch, /ingest/cleanup, /memory/ingest, and
# GET /ingest/pipeline/status require X-Deltai-Ingest-Key or Authorization: Bearer.
DELTAI_INGEST_API_KEY = os.getenv("DELTAI_INGEST_API_KEY", "").strip()
# Optional: when set, POST /chat requires X-Deltai-Chat-Key or Authorization: Bearer (same pattern as ingest).
DELTAI_CHAT_API_KEY = os.getenv("DELTAI_CHAT_API_KEY", "").strip()


def _cors_allow_origins() -> list[str]:
    """Comma-separated allowlist; unset defaults to '*' (localhost dev). Set for LAN or tunnel exposure."""
    raw = os.getenv("DELTAI_CORS_ORIGINS", "").strip()
    if not raw:
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


def require_ingest_api_key(
    _x_deltai_ingest_key: str | None = Header(default=None, alias="X-Deltai-Ingest-Key"),
    authorization: str | None = Header(default=None),
) -> None:
    if not DELTAI_INGEST_API_KEY:
        return
    if _x_deltai_ingest_key and _x_deltai_ingest_key.strip() == DELTAI_INGEST_API_KEY:
        return
    if authorization:
        a = authorization.strip()
        if a.lower().startswith("bearer ") and a[7:].strip() == DELTAI_INGEST_API_KEY:
            return
    raise HTTPException(
        status_code=401,
        detail="Ingest key required: X-Deltai-Ingest-Key or Authorization: Bearer (match DELTAI_INGEST_API_KEY)",
    )


def require_chat_api_key(
    _x_deltai_chat_key: str | None = Header(default=None, alias="X-Deltai-Chat-Key"),
    authorization: str | None = Header(default=None),
) -> None:
    if not DELTAI_CHAT_API_KEY:
        return
    if _x_deltai_chat_key and _x_deltai_chat_key.strip() == DELTAI_CHAT_API_KEY:
        return
    if authorization:
        a = authorization.strip()
        if a.lower().startswith("bearer ") and a[7:].strip() == DELTAI_CHAT_API_KEY:
            return
    raise HTTPException(
        status_code=401,
        detail="Chat key required: X-Deltai-Chat-Key or Authorization: Bearer (match DELTAI_CHAT_API_KEY)",
    )


# project/ (parent of deltai_api/) — same layout as when this lived in main.py
_HERE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# ── EMERGENCY BACKUP STATE ────────────────────────────────────────────

_backup_health_status: dict = {}  # {model: {"healthy": bool, "last_check": float}}
_backup_last_activated: float = 0  # timestamp of last emergency activation

# ── WEBSOCKET ALERT SYSTEM ──────────────────────────────────────────
_alert_clients: list[WebSocket] = []
_recent_alerts: deque = deque(maxlen=20)


async def _broadcast_alert(alert: dict):
    """Send an alert to all connected WebSocket clients."""
    dead = []
    msg = json.dumps(alert)
    for ws in _alert_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        try:
            _alert_clients.remove(ws)
        except ValueError:
            pass


# ── HEALTH EVENT BUS ─────────────────────────────────────────────────
# Typed events for subsystem state changes, pushed to frontend via WebSocket.
_health_clients: list[WebSocket] = []
_health_events: deque = deque(maxlen=100)


def _record_health_event(event_type: str, data: dict) -> dict:
    """Sync: record event in deque + log. Safe to call from sync contexts (circuit breaker)."""
    event = {"type": event_type, "ts": _time.time(), **data}
    _health_events.append(event)
    logger.info(f"Health event: {event_type} — {data}")
    return event


async def _emit_health_event(event_type: str, data: dict):
    """Async: record + broadcast to WebSocket health clients."""
    event = _record_health_event(event_type, data)
    dead = []
    for ws in _health_clients:
        try:
            await ws.send_json(event)
        except Exception:
            dead.append(ws)
    for ws in dead:
        try:
            _health_clients.remove(ws)
        except ValueError:
            pass


# ── IDLE TRACKING (for self-heal) ─────────────────────────────────────
_last_chat_start: float = 0.0
_last_chat_end: float = 0.0


def _is_idle() -> bool:
    """Check if deltai is idle (no active chat, 30s breathing room after last chat)."""
    if _last_chat_start > _last_chat_end:
        return False  # chat in progress
    if _time.time() - _last_chat_end < 30:
        return False  # recently finished
    return True


# ── CONVERSATION HISTORY ─────────────────────────────────────────────
HISTORY_MAX_TURNS = int(os.getenv("CONVERSATION_HISTORY_MAX", "10"))
_conversation_history: list[dict] = []


def _append_to_history(user_message: str, assistant_response: str, chat_metadata: dict = None):
    """
    Store a clean user-assistant exchange. Skip if either is empty.
    Optional chat_metadata: {tier, domain, tool_calls, tool_results, react_used, model, latency_ms}
    """
    global _last_chat_end
    _last_chat_end = _time.time()
    if not user_message.strip() or not assistant_response.strip():
        return
    _conversation_history.append({"role": "user", "content": user_message})
    _conversation_history.append({"role": "assistant", "content": assistant_response})
    max_msgs = HISTORY_MAX_TURNS * 2
    if len(_conversation_history) > max_msgs:
        del _conversation_history[:-max_msgs]
    # Persist to SQLite
    try:
        save_history_pair(user_message, assistant_response)
        trim_history(HISTORY_MAX_TURNS)
    except Exception as e:
        logger.warning("Failed to persist history [%s]", type(e).__name__)

    # ── Quality scoring + smart capture ──
    metadata = chat_metadata or {}
    quality_result = None
    try:
        from quality import score_response

        quality_result = score_response(user_message, assistant_response, metadata)
        # Persist quality score
        save_quality_score(
            query_text=user_message,
            response_preview=assistant_response[:200],
            score=quality_result["score"],
            signals_json=json.dumps(quality_result["signals"]),
            tier=metadata.get("tier", 1),
            domain=metadata.get("domain", "general"),
        )
    except Exception as e:
        logger.debug("Quality scoring failed [%s]", type(e).__name__)

    # ── Routing feedback ──
    if metadata.get("model") and quality_result:
        try:
            import hashlib

            qhash = hashlib.sha256(user_message.strip().lower().encode()).hexdigest()
            save_routing_feedback(
                query_hash=qhash,
                classified_tier=metadata.get("tier", 1),
                actual_model=metadata.get("model", ""),
                domain=metadata.get("domain", "general"),
                quality_score=quality_result["score"],
                latency_ms=metadata.get("latency_ms", 0),
                tool_calls_count=len(metadata.get("tool_calls", [])),
            )
        except Exception as e:
            logger.debug("Routing feedback save failed [%s]", type(e).__name__)

    # ── Knowledge gap detection ──
    if quality_result and quality_result["score"] < 0.3:
        try:
            save_knowledge_gap(
                query_text=user_message,
                domain=metadata.get("domain", "general"),
                quality_score=quality_result["score"],
                gap_type="low_quality",
            )
        except Exception:
            pass

    # ── Smart auto-capture (replaces basic auto_capture) ──
    if TRAINING_AVAILABLE:
        try:
            from training import smart_auto_capture

            smart_auto_capture(
                "deltai-auto",
                user_message,
                assistant_response,
                quality_score=quality_result["score"] if quality_result else 0.5,
                metadata=metadata,
            )
        except Exception:
            # Fall back to basic auto_capture
            try:
                auto_capture("deltai-auto", user_message, assistant_response)
            except Exception:
                pass

    try:
        from delta.storage.reports import write_chat_turn_report

        write_chat_turn_report(
            user_message=user_message,
            assistant_response=assistant_response,
            chat_metadata=metadata,
            status="ok",
        )
    except ImportError:
        pass


def _get_history() -> list[dict]:
    """Return a copy of conversation history for injection into message arrays."""
    return list(_conversation_history)


# ── SMART HISTORY COMPRESSION ────────────────────────────────────────────
_SMART_HISTORY_ENABLED = os.getenv("SMART_HISTORY_ENABLED", "true").lower() in ("true", "1", "yes")


def _compress_turn(content: str) -> str:
    """
    Compress a conversation turn to key information.
    Extracts sentences with numbers, proper nouns, or conclusions.
    """
    import re as _re

    sentences = _re.split(r"[.!?]\s+", content)
    key_sentences = []
    for s in sentences:
        s = s.strip()
        if not s or len(s) < 10:
            continue
        # Keep sentences with numbers, units, or technical terms
        has_numbers = bool(_re.search(r"\d+\.?\d*\s*(?:MB|GB|ms|s|°C|%|rpm|kg|N|Pa)", s))
        has_code = "`" in s or "```" in s
        has_conclusion = any(
            w in s.lower()
            for w in [
                "result",
                "therefore",
                "conclusion",
                "answer",
                "solution",
                "should",
                "recommend",
                "optimal",
                "best",
            ]
        )
        if has_numbers or has_code or has_conclusion:
            key_sentences.append(s)

    if key_sentences:
        return ". ".join(key_sentences[:3]) + "."
    # Fallback: first 100 chars
    return content[:100] + "..." if len(content) > 100 else content


def _summarize_turn(content: str) -> str:
    """One-line summary of a conversation turn."""
    # Extract the first meaningful sentence
    content = content.strip()
    first_sentence = content.split(".")[0].split("?")[0].split("!")[0]
    if len(first_sentence) > 80:
        first_sentence = first_sentence[:77] + "..."
    return first_sentence


def _get_smart_history(max_tokens: int = None) -> list[dict]:
    """
    Return conversation history with intelligent compression.
    - Last 3 turns: full content
    - Turns 4-7: compressed to key facts
    - Turns 8-10: one-line summaries
    """
    if not _SMART_HISTORY_ENABLED:
        return list(_conversation_history)

    history = list(_conversation_history)
    if len(history) <= 6:  # 3 turns = 6 messages
        return history

    result = []
    n_msgs = len(history)
    n_turns = n_msgs // 2

    for i in range(0, n_msgs, 2):
        turn_idx = i // 2  # 0-based turn index
        turns_from_end = n_turns - turn_idx

        user_msg = history[i]
        assistant_msg = history[i + 1] if i + 1 < n_msgs else None

        if turns_from_end <= 3:
            # Recent: keep full
            result.append(user_msg)
            if assistant_msg:
                result.append(assistant_msg)
        elif turns_from_end <= 7:
            # Middle: compress
            result.append({"role": "user", "content": _compress_turn(user_msg["content"])})
            if assistant_msg:
                result.append(
                    {"role": "assistant", "content": _compress_turn(assistant_msg["content"])}
                )
        else:
            # Old: summarize
            result.append({"role": "user", "content": _summarize_turn(user_msg["content"])})
            if assistant_msg:
                result.append(
                    {"role": "assistant", "content": _summarize_turn(assistant_msg["content"])}
                )

    return result


async def _try_ollama_inference(
    client: httpx.AsyncClient,
    model: str,
    messages: list,
    tools: list | None = None,
    num_gpu: int | None = None,
    num_ctx: int | None = None,
) -> tuple[dict | None, str | None]:
    """
    Attempt a single Ollama inference call with circuit breaker protection.
    Returns (data, None) on success or (None, error_string) on failure.
    Supports num_gpu (partial GPU offloading) and num_ctx (dynamic context window).
    """
    # Circuit breaker: don't hammer Ollama when it's down
    if not _cb_check():
        return (
            None,
            f"Circuit breaker open — Ollama unreachable (backoff {_circuit_breaker['backoff_sec']}s)",
        )

    payload = {"model": model, "messages": messages, "stream": False}
    if tools:
        payload["tools"] = tools
    # Dynamic GPU layer offloading + context window sizing
    options = {}
    if num_gpu is not None:
        options["num_gpu"] = num_gpu
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    temp_raw = os.getenv("DELTAI_OLLAMA_TEMPERATURE", "0.2").strip()
    top_p_raw = os.getenv("DELTAI_OLLAMA_TOP_P", "0.95").strip()
    try:
        if temp_raw:
            options["temperature"] = float(temp_raw)
    except ValueError:
        pass
    try:
        if top_p_raw:
            options["top_p"] = float(top_p_raw)
    except ValueError:
        pass
    if options:
        payload["options"] = options
    try:
        resp = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        if resp.status_code != 200:
            _cb_failure()
            return None, f"HTTP {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
        if "error" in data:
            # Model errors aren't Ollama connectivity failures
            _cb_success()
            return None, data["error"]
        _cb_success()
        return data, None
    except httpx.TimeoutException:
        _cb_failure()
        return None, "Ollama request timed out"
    except (httpx.ConnectError, httpx.ConnectTimeout):
        _cb_failure()
        return None, "Cannot connect to Ollama"
    except Exception as e:
        _cb_failure()
        safe_errors.log_exception(logger, "Ollama inference request failed", e)
        return None, safe_errors.public_error_detail(e)


async def _inference_with_emergency_fallback(
    client: httpx.AsyncClient,
    model: str,
    messages: list,
    tools: list | None,
    backup_model: str | None,
    num_gpu: int | None = None,
    num_ctx: int | None = None,
) -> tuple[dict | None, str | None, str, bool]:
    """
    Try primary model with retries, then emergency backup as last resort.

    Returns (data, error, used_model, is_emergency).
    """
    # ── Attempt primary (with retries) ──
    last_error = None
    for attempt in range(1 + BACKUP_MAX_RETRIES):
        data, err = await _try_ollama_inference(
            client, model, messages, tools, num_gpu=num_gpu, num_ctx=num_ctx
        )
        if data is not None:
            return data, None, model, False
        last_error = err
        if attempt < BACKUP_MAX_RETRIES:
            logger.warning(
                f"Primary model {model} failed (attempt {attempt + 1}): {err} — retrying in 3s"
            )
            await asyncio.sleep(3)

    # ── Primary exhausted — engage emergency backup ──
    # Backups run with default settings (no partial offload)
    if backup_model:
        logger.error(
            f"PRIMARY MODEL DOWN: {model} failed {1 + BACKUP_MAX_RETRIES} attempts. "
            f"Engaging emergency backup: {backup_model}"
        )
        data, err = await _try_ollama_inference(client, backup_model, messages, tools)
        if data is not None:
            global _backup_last_activated
            _backup_last_activated = _time.time()
            return data, None, backup_model, True
        # Backup also failed — try one more in the chain
        second_backup = get_backup_model(backup_model)
        if second_backup:
            logger.error(f"Backup {backup_model} also failed. Last resort: {second_backup}")
            data, err2 = await _try_ollama_inference(client, second_backup, messages, tools)
            if data is not None:
                _backup_last_activated = _time.time()
                return data, None, second_backup, True
        return None, f"All models failed. Primary: {last_error}", model, False

    return None, last_error, model, False


# ── ReAct REASONING LOOP ──────────────────────────────────────────────
# Structured Think→Act→Observe→Respond loop for complex multi-step queries.
# Activated for Tier 2/3 complexity when using local models.

_REACT_MAX_ITERATIONS = int(os.getenv("REACT_MAX_ITERATIONS", "3"))
_REACT_ENABLED = os.getenv("REACT_ENABLED", "true").lower() in ("true", "1", "yes")

_REACT_ALLOW_CLARIFY = os.getenv("REACT_ALLOW_CLARIFY", "true").lower() in ("true", "1", "yes")

def _is_react_eligible(decision) -> bool:
    """Check if a query should use the ReAct reasoning loop."""
    if not _REACT_ENABLED:
        return False
    if decision.backend != "ollama":
        return False
    # Only for complex queries (Tier 2/3) on local models
    if decision.tier < 2:
        return False
    return True


async def _react_reasoning_loop(
    client,
    model: str,
    user_message: str,
    rag_context: str,
    history: list,
    tools: list,
    execute_fn,
    num_gpu: int | None = None,
    num_ctx: int | None = None,
) -> tuple[str, list]:
    """
    Execute a ReAct (Think-Act-Observe) reasoning loop.

    Returns (final_response_text, tool_events_for_streaming).
    """
    events = []  # list of dicts for streaming to frontend

    # ── Retrieve similar past reasoning traces ──
    prior_context = ""
    if os.getenv("REASONING_TRACE_ENABLED", "true").lower() in ("true", "1", "yes"):
        try:
            from memory import get_embeddings

            query_emb = get_embeddings([user_message])[0]
            query_emb_bytes = _serialize_embedding(query_emb)
            similar = find_similar_traces(query_emb_bytes, n=3)
            if similar:
                parts = ["[PRIOR REASONING — similar problems you solved before]"]
                for trace in similar:
                    parts.append(
                        f"Query: {trace['query_text'][:100]}\n"
                        f"Domain: {trace['domain']}\n"
                        f"Tools used: {trace['tool_sequence']}\n"
                        f"Result: {trace['final_summary']}"
                    )
                parts.append("[END PRIOR REASONING]\n")
                prior_context = "\n\n".join(parts)
        except Exception as e:
            logger.debug("Trace retrieval failed [%s]", type(e).__name__)

    react_system = build_react_system_prompt(_REACT_MAX_ITERATIONS)
    context_prefix = f"{rag_context}\n" if rag_context else ""

    messages = (
        [{"role": "system", "content": react_system}]
        + history
        + [{"role": "user", "content": f"{prior_context}{context_prefix}{user_message}"}]
    )

    for _iteration in range(_REACT_MAX_ITERATIONS):
        data, err = await _try_ollama_inference(
            client, model, messages, tools, num_gpu=num_gpu, num_ctx=num_ctx
        )
        if data is None:
            return f"Reasoning error: {err}", events

        msg = data.get("message", {})
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls")

        # Extract confidence level if present
        confidence = "unknown"
        if "CONFIDENCE: HIGH" in content:
            confidence = "high"
        elif "CONFIDENCE: MEDIUM" in content:
            confidence = "medium"
        elif "CONFIDENCE: LOW" in content:
            confidence = "low"

        # Check if model reached a FINAL answer
        if "FINAL:" in content:
            final_text = content.split("FINAL:", 1)[1].strip()
            _save_react_trace(user_message, final_text, content, events, confidence=confidence)
            return final_text, events

        # Check if model needs clarification from user
        if "CLARIFY:" in content and _REACT_ALLOW_CLARIFY:
            clarify_text = content.split("CLARIFY:", 1)[1].strip()
            events.append({"t": "clarify", "c": clarify_text})
            # Return clarification request — frontend will handle user input
            _save_react_trace(
                user_message, "", content, events, confidence=confidence, success=False
            )
            return f"[CONFIDENCE: LOW] I need more information: {clarify_text}", events

        # Process tool calls if any
        if tool_calls:
            messages.append({"role": "assistant", "content": content})
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "unknown")
                args = fn.get("arguments", {})
                events.append({"t": "tool", "n": name, "a": args})

                try:
                    result = execute_fn(name, args)
                    result_str = str(result)[:4000]
                except Exception as e:
                    safe_errors.log_exception(logger, "ReAct tool execution failed", e)
                    result_str = "Error: tool execution failed"

                events.append({"t": "result", "n": name, "s": result_str[:200]})
                messages.append({"role": "tool", "content": result_str})
        else:
            # No tool calls and no FINAL — model is thinking
            messages.append({"role": "assistant", "content": content})
            # Nudge toward conclusion
            messages.append(
                {
                    "role": "user",
                    "content": "Continue reasoning. If you have enough information, respond with FINAL: followed by your answer.",
                }
            )

    # Max iterations reached — extract best answer from last response
    if content:
        # Try to find any FINAL or just use the last content
        if "FINAL:" in content:
            final_text = content.split("FINAL:", 1)[1].strip()
            _save_react_trace(user_message, final_text, content, events)
            return final_text, events
        _save_react_trace(user_message, content, content, events)
        return content, events
    _save_react_trace(
        user_message, "Reasoning loop completed without a clear answer.", content or "", events
    )
    return "Reasoning loop completed without a clear answer.", events


def _save_react_trace(
    user_message: str,
    final_text: str,
    content: str,
    events: list,
    confidence: str = "unknown",
    success: bool = None,
):
    """Save a reasoning trace after a ReAct loop completes."""
    if os.getenv("REASONING_TRACE_ENABLED", "true").lower() not in ("true", "1", "yes"):
        return
    try:
        from memory import get_embeddings

        query_emb = get_embeddings([user_message])[0]
        query_emb_bytes = _serialize_embedding(query_emb)
        tool_seq = ",".join(ev.get("n", "") for ev in events if ev.get("t") == "tool")
        was_success = (
            success if success is not None else ("FINAL:" in (content or "") or bool(final_text))
        )
        save_reasoning_trace(
            query_text=user_message,
            domain=None,
            steps_json=json.dumps(events),
            final_summary=final_text[:500] if final_text else "",
            tool_sequence=tool_seq,
            success=was_success,
            confidence=confidence,
            embedding=query_emb_bytes,
        )
    except Exception as e:
        logger.debug("Trace save failed [%s]", type(e).__name__)


async def _backup_health_loop():
    """
    Hourly background task: ping each backup model with 1-token generation
    to verify it remains functional. Dormant infrastructure check.
    """
    interval = int(os.getenv("BACKUP_HEALTH_INTERVAL_SEC", "3600"))
    if os.getenv("BACKUP_ENABLED", "true").lower() not in ("true", "1", "yes"):
        logger.info("Backup system disabled — health loop not started")
        return

    while True:
        await asyncio.sleep(interval)
        for key in ("DELTAI_BACKUP_STRONG_MODEL", "DELTAI_BACKUP_MODEL"):
            model = os.getenv(key, "").strip()
            if not model:
                continue
            healthy = await check_model_health(model)
            _backup_health_status[model] = {
                "healthy": healthy,
                "last_check": _time.time(),
            }
            if healthy:
                logger.info(f"Backup health OK: {model}")
            else:
                logger.warning(f"BACKUP HEALTH FAIL: {model} — emergency generator offline")
            await asyncio.sleep(15)  # pause between pings to avoid VRAM contention


# ── RESOURCE SELF-MANAGER ──────────────────────────────────────────────
# Background task that monitors VRAM, model lifecycle, and subsystem health.
# Takes automatic corrective actions when resources are constrained.

_resource_state = {
    "last_vram_action": 0.0,  # timestamp of last VRAM management action
    "last_recovery": 0.0,  # timestamp of last auto-recovery attempt
    "vram_warnings": 0,  # consecutive low-VRAM readings
    "ollama_failures": 0,  # consecutive Ollama connectivity failures
    "watcher_restarts": 0,  # number of watcher auto-restarts this session
    "actions_taken": [],  # log of automatic actions (last 50)
    "last_sim_state": False,  # previous sim running state (for transition detection)
    "sim_stop_detected_at": 0.0,  # when sim stop was first detected
    "pending_14b_preload": False,  # waiting to preload 14B after sim stop
    "vram_history": [],  # sliding window of (timestamp, vram_free_mb) for prediction
    "priority_lowered": False,  # whether OS process priority is currently lowered
}
_RESOURCE_CHECK_INTERVAL = 30  # seconds between resource checks
_VRAM_CRITICAL_MB = 1500  # below this, aggressively free VRAM
_VRAM_WARN_MB = 3000  # below this, start monitoring closely
_MAX_WATCHER_RESTARTS = 5  # don't restart watcher more than this
_OLLAMA_FAILURE_THRESHOLD = 3  # consecutive failures before auto-restart attempt
_RESOURCE_ACTION_COOLDOWN = 60  # seconds between automatic VRAM actions
_VRAM_HISTORY_WINDOW = 60  # seconds of VRAM history to keep for prediction
_VRAM_DECLINE_RATE_THRESHOLD = 100  # MB/s — preemptive unload if declining faster
_GPU_TEMP_THROTTLE = 80  # celsius — switch to smaller model above this


def _log_resource_action(action: str):
    """Record an automatic resource management action."""
    entry = {"action": action, "ts": _time.time()}
    _resource_state["actions_taken"].append(entry)
    if len(_resource_state["actions_taken"]) > 50:
        _resource_state["actions_taken"] = _resource_state["actions_taken"][-50:]
    logger.info(f"Resource manager: {action}")


def _adjust_process_priority(lower: bool):
    """
    Adjust OS-level process priority for deltai and Ollama.
    When a GPU focus workload is active or VRAM pressure is high, lower priority so foreground apps get CPU/IO first.
    """
    if _resource_state["priority_lowered"] == lower:
        return  # already in desired state

    try:
        target = psutil.BELOW_NORMAL_PRIORITY_CLASS if lower else psutil.NORMAL_PRIORITY_CLASS
        label = "BELOW_NORMAL" if lower else "NORMAL"

        # Adjust own process
        own = psutil.Process()
        own.nice(target)

        # Adjust Ollama process(es)
        for proc in psutil.process_iter(["name", "pid"]):
            try:
                if proc.info["name"] and "ollama" in proc.info["name"].lower():
                    proc.nice(target)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        _resource_state["priority_lowered"] = lower
        _log_resource_action(f"Process priority → {label}")
    except Exception as e:
        logger.debug("Priority adjustment failed [%s]", type(e).__name__)


def _record_vram_reading(vram_free: int):
    """Record a VRAM reading for predictive trend analysis."""
    now = _time.time()
    history = _resource_state["vram_history"]
    history.append((now, vram_free))
    # Trim to window
    cutoff = now - _VRAM_HISTORY_WINDOW
    _resource_state["vram_history"] = [(t, v) for t, v in history if t >= cutoff]


def _predict_vram_decline() -> float:
    """
    Calculate VRAM decline rate (MB/s) from recent history.
    Positive = declining (losing VRAM), negative = recovering.
    Returns 0.0 if insufficient data.
    """
    history = _resource_state["vram_history"]
    if len(history) < 3:
        return 0.0
    # Use first and last readings for slope
    t0, v0 = history[0]
    t1, v1 = history[-1]
    dt = t1 - t0
    if dt < 5:  # need at least 5s of data
        return 0.0
    # Positive = VRAM is decreasing (bad)
    return (v0 - v1) / dt


def _get_gpu_temp() -> int:
    """Get GPU temperature in celsius, or 0 if unavailable."""
    try:
        import pynvml

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    except Exception:
        return 0


async def _resource_manager_loop():
    """
    Background resource manager. Monitors VRAM, Ollama, watcher, and
    takes automatic corrective actions:

    1. VRAM management: Unloads idle models when VRAM is critical
    2. Model lifecycle: Preloads appropriate model for current tier
    3. Ollama health: Detects failures and attempts restart
    4. Watcher recovery: Restarts file watcher if it dies
    5. TTL cleanup: Periodic cleanup of expired RAG entries
    """
    await asyncio.sleep(15)  # let startup finish
    logger.info("Resource self-manager started")

    while True:
        try:
            await asyncio.sleep(_RESOURCE_CHECK_INTERVAL)
            now = _time.time()

            # ── 0. PROCESS PRIORITY + THERMAL MANAGEMENT ──
            vram_free = get_vram_free_mb()
            sim_active = is_sim_running()
            _record_vram_reading(vram_free)

            # Lower process priority when sim running or VRAM under pressure
            should_lower = sim_active or vram_free < _VRAM_WARN_MB
            _adjust_process_priority(should_lower)

            # Thermal-aware: if GPU is hot, prefer smaller model / CPU offload
            gpu_temp = _get_gpu_temp()
            if gpu_temp > _GPU_TEMP_THROTTLE and not sim_active:
                if now - _resource_state["last_vram_action"] > _RESOURCE_ACTION_COOLDOWN:
                    try:
                        async with httpx.AsyncClient(timeout=5) as client:
                            resp = await client.get(f"{OLLAMA_URL}/api/ps")
                            if resp.status_code == 200:
                                for m in resp.json().get("models", []):
                                    if "14b" in m.get("name", "").lower():
                                        await client.post(
                                            f"{OLLAMA_URL}/api/generate",
                                            json={
                                                "model": m["name"],
                                                "prompt": "",
                                                "keep_alive": 0,
                                            },
                                            timeout=10,
                                        )
                                        _log_resource_action(
                                            f"Unloaded {m['name']} (GPU temp {gpu_temp}°C)"
                                        )
                                        _resource_state["last_vram_action"] = now
                    except Exception:
                        pass

            # ── 0b. PREDICTIVE VRAM — preemptive action on rapid decline ──
            decline_rate = _predict_vram_decline()
            if (
                decline_rate > _VRAM_DECLINE_RATE_THRESHOLD
                and vram_free < _VRAM_WARN_MB * 1.5
                and now - _resource_state["last_vram_action"] > _RESOURCE_ACTION_COOLDOWN
            ):
                try:
                    async with httpx.AsyncClient(timeout=5) as client:
                        resp = await client.get(f"{OLLAMA_URL}/api/ps")
                        if resp.status_code == 200:
                            for m in resp.json().get("models", []):
                                if "14b" in m.get("name", "").lower():
                                    await client.post(
                                        f"{OLLAMA_URL}/api/generate",
                                        json={"model": m["name"], "prompt": "", "keep_alive": 0},
                                        timeout=10,
                                    )
                                    _log_resource_action(
                                        f"Preemptive unload {m['name']} "
                                        f"(VRAM declining {decline_rate:.0f}MB/s, {vram_free}MB free)"
                                    )
                                    _resource_state["last_vram_action"] = now
                except Exception:
                    pass

            # ── 1. VRAM MANAGEMENT ──
            if vram_free < _VRAM_CRITICAL_MB:
                _resource_state["vram_warnings"] += 1
                # Critical VRAM — aggressive action after 2 consecutive readings
                if (
                    _resource_state["vram_warnings"] >= 2
                    and now - _resource_state["last_vram_action"] > _RESOURCE_ACTION_COOLDOWN
                ):
                    # Check what's loaded and unload non-essential models
                    try:
                        async with httpx.AsyncClient(timeout=5) as client:
                            resp = await client.get(f"{OLLAMA_URL}/api/ps")
                            if resp.status_code == 200:
                                loaded = resp.json().get("models", [])
                                # During sim, unload the 14B if it's loaded
                                for m in loaded:
                                    name = m.get("name", "").split(":")[0]
                                    if sim_active and "14b" in name:
                                        await client.post(
                                            f"{OLLAMA_URL}/api/generate",
                                            json={
                                                "model": m["name"],
                                                "prompt": "",
                                                "keep_alive": 0,
                                            },
                                            timeout=10,
                                        )
                                        _log_resource_action(
                                            f"Unloaded {m['name']} (VRAM critical: {vram_free}MB, sim active)"
                                        )
                                    elif vram_free < 1000 and len(loaded) > 1:
                                        # Extreme pressure — unload everything except smallest
                                        sizes = [
                                            (
                                                m2.get("name", ""),
                                                m2.get("size_vram", m2.get("size", 0)),
                                            )
                                            for m2 in loaded
                                        ]
                                        sizes.sort(key=lambda x: x[1], reverse=True)
                                        for model_name, _ in sizes[:-1]:  # keep smallest
                                            await client.post(
                                                f"{OLLAMA_URL}/api/generate",
                                                json={
                                                    "model": model_name,
                                                    "prompt": "",
                                                    "keep_alive": 0,
                                                },
                                                timeout=10,
                                            )
                                            _log_resource_action(
                                                f"Unloaded {model_name} (VRAM extreme: {vram_free}MB)"
                                            )
                                        break
                                _resource_state["last_vram_action"] = now
                    except Exception as e:
                        logger.debug("VRAM management failed [%s]", type(e).__name__)

            elif vram_free < _VRAM_WARN_MB:
                _resource_state["vram_warnings"] = max(1, _resource_state["vram_warnings"])
            else:
                _resource_state["vram_warnings"] = 0

            # ── 2. MODEL LIFECYCLE (proactive sim transitions) ──
            prev_sim = _resource_state["last_sim_state"]
            _resource_state["last_sim_state"] = sim_active

            # Sim just STARTED (False -> True)
            if sim_active and not prev_sim:
                _log_resource_action("Sim started — proactive model swap")
                await _emit_health_event("sim_state_changed", {"running": True})
                try:
                    async with httpx.AsyncClient(timeout=10) as mc:
                        ps_resp = await mc.get(f"{OLLAMA_URL}/api/ps")
                        if ps_resp.status_code == 200:
                            for m in ps_resp.json().get("models", []):
                                if "14b" in m.get("name", "").lower():
                                    await mc.post(
                                        f"{OLLAMA_URL}/api/generate",
                                        json={"model": m["name"], "prompt": "", "keep_alive": 0},
                                        timeout=10,
                                    )
                                    _log_resource_action(f"Unloaded {m['name']} (sim started)")
                                    await _emit_health_event(
                                        "model_unloaded",
                                        {"model": m["name"], "reason": "sim_start"},
                                    )
                        # Preload 3B if VRAM allows
                        vram_now = get_vram_free_mb()
                        sim_model = os.getenv(
                            "DELTAI_SIM_MODEL", os.getenv("DELTAI_MODEL", "deltai-qwen3b")
                        )
                        if vram_now >= 2500:
                            await mc.post(
                                f"{OLLAMA_URL}/api/generate",
                                json={
                                    "model": sim_model,
                                    "prompt": "ping",
                                    "keep_alive": "5m",
                                    "options": {"num_predict": 1},
                                },
                                timeout=60,
                            )
                            _log_resource_action(
                                f"Preloaded {sim_model} (sim started, {vram_now}MB free)"
                            )
                            await _emit_health_event(
                                "model_loaded", {"model": sim_model, "reason": "sim_start"}
                            )
                except Exception as e:
                    logger.debug("Sim-start model swap failed [%s]", type(e).__name__)
                _resource_state["pending_14b_preload"] = False

            # Sim just STOPPED (True -> False)
            elif not sim_active and prev_sim:
                _log_resource_action("Sim stopped — scheduling 14B preload")
                await _emit_health_event("sim_state_changed", {"running": False})
                _resource_state["sim_stop_detected_at"] = now
                _resource_state["pending_14b_preload"] = True

            # Pending 14B preload (~30s after sim stop)
            if _resource_state["pending_14b_preload"]:
                elapsed_since_stop = now - _resource_state["sim_stop_detected_at"]
                if elapsed_since_stop >= 30:
                    _resource_state["pending_14b_preload"] = False
                    vram_now = get_vram_free_mb()
                    strong_model = os.getenv("DELTAI_STRONG_MODEL", "deltai-qwen14b")
                    if vram_now >= 9000:
                        try:
                            async with httpx.AsyncClient(timeout=60) as mc:
                                await mc.post(
                                    f"{OLLAMA_URL}/api/generate",
                                    json={
                                        "model": strong_model,
                                        "prompt": "ping",
                                        "keep_alive": "5m",
                                        "options": {"num_predict": 1},
                                    },
                                    timeout=60,
                                )
                                _log_resource_action(
                                    f"Preloaded {strong_model} (sim stopped, {vram_now}MB free)"
                                )
                                await _emit_health_event(
                                    "model_loaded", {"model": strong_model, "reason": "sim_stop"}
                                )
                        except Exception as e:
                            logger.debug("Post-sim 14B preload failed [%s]", type(e).__name__)
                    else:
                        _log_resource_action(f"Skipped 14B preload — only {vram_now}MB VRAM free")

            # ── 3. OLLAMA HEALTH ──
            try:
                async with httpx.AsyncClient(timeout=3) as client:
                    resp = await client.get(f"{OLLAMA_URL}/api/tags")
                    if resp.status_code == 200:
                        _resource_state["ollama_failures"] = 0
                    else:
                        _resource_state["ollama_failures"] += 1
            except Exception:
                _resource_state["ollama_failures"] += 1

            if _resource_state["ollama_failures"] >= _OLLAMA_FAILURE_THRESHOLD:
                if now - _resource_state["last_recovery"] > 120:  # 2min cooldown
                    _log_resource_action(
                        f"Ollama unreachable ({_resource_state['ollama_failures']} failures), attempting restart"
                    )
                    try:
                        import subprocess

                        subprocess.Popen(
                            ["ollama", "serve"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        _resource_state["last_recovery"] = now
                        _resource_state["ollama_failures"] = 0
                        await asyncio.sleep(5)  # give it time to start
                    except Exception as e:
                        safe_errors.log_exception(logger, "Ollama auto-restart failed", e)

            # ── 4. WATCHER RECOVERY ──
            if RAG_AVAILABLE:
                try:
                    if (
                        not watcher_running()
                        and _resource_state["watcher_restarts"] < _MAX_WATCHER_RESTARTS
                    ):
                        _log_resource_action("File watcher died, restarting")
                        stop_watcher()
                        start_watcher()
                        _resource_state["watcher_restarts"] += 1
                except Exception as e:
                    logger.debug("Watcher recovery check failed [%s]", type(e).__name__)

            # ── 5. PERIODIC TTL CLEANUP ──
            if RAG_AVAILABLE:
                try:
                    cleanup_expired()
                except Exception:
                    pass

            # ── 7. REASONING TRACE PRUNING (every ~30 min) ──
            if int(now) % 1800 < _RESOURCE_CHECK_INTERVAL:
                try:
                    prune_old_traces(
                        max_age_days=30, max_count=int(os.getenv("REASONING_TRACE_MAX", "500"))
                    )
                except Exception:
                    pass

            # ── 6. HIERARCHICAL MEMORY COMPACTION (every ~10 min) ──
            if RAG_AVAILABLE and int(now) % 600 < _RESOURCE_CHECK_INTERVAL:
                try:
                    result = compact_warm_to_cold()
                    if result.get("demoted", 0) > 0:
                        _log_resource_action(f"Cold compaction: {result['demoted']} chunks demoted")
                except Exception:
                    pass

        except asyncio.CancelledError:
            raise
        except Exception as e:
            safe_errors.log_exception(logger, "Resource manager error", e)
            await asyncio.sleep(10)


# ── AI-DRIVEN SELF-HEAL LOOP ──────────────────────────────────────────
# Periodically runs diagnostics, has the LLM reason about issues, and
# executes repairs autonomously. Only runs when idle + not in session.

_SELF_HEAL_ENABLED = os.getenv("SELF_HEAL_ENABLED", "true").lower() in ("true", "1", "yes")
_SELF_HEAL_INTERVAL = int(os.getenv("SELF_HEAL_INTERVAL_SEC", "300"))  # 5 minutes
_SELF_HEAL_MODEL = os.getenv("DELTAI_MODEL", "deltai-qwen3b")

_SELF_HEAL_SYSTEM_PROMPT = f"""You are deltai's internal diagnostics AI. You receive a system diagnostics report and must decide if any repairs are needed.

RULES:
- Only suggest repairs if there are actual issues (DOWN, ERROR, STOPPED, WARNING, DEGRADED)
- Available repairs: restart_watcher, clear_vram, reindex_knowledge, check_ollama
- If no issues found, respond with exactly: NO_ACTION
- If repair needed, respond with exactly: REPAIR:<repair_name>
- Only suggest ONE repair per cycle (most critical first)
- Never suggest clear_vram during an active GPU focus session
- Never suggest repairs for cosmetic warnings
- {protocol_antifabrication_reminder()}

Examples:
- Diagnostics show "Watcher: STOPPED" -> REPAIR:restart_watcher
- Diagnostics show "Ollama: DOWN" -> REPAIR:check_ollama
- Diagnostics show everything nominal -> NO_ACTION
"""


async def _ai_self_heal_loop():
    """
    AI-driven self-healing loop. Periodically runs diagnostics,
    has the LLM reason about results, and executes repairs.
    """
    if not _SELF_HEAL_ENABLED:
        logger.info("Self-heal loop disabled")
        return

    await asyncio.sleep(60)  # let startup + first resource manager cycle finish
    logger.info("AI self-heal loop started")

    while True:
        try:
            await asyncio.sleep(_SELF_HEAL_INTERVAL)

            # Skip if not idle
            if not _is_idle():
                logger.debug("Self-heal: skipping (chat active)")
                continue

            # Skip if circuit breaker is open (Ollama down — can't run LLM)
            if not _cb_check():
                logger.debug("Self-heal: skipping (circuit breaker open)")
                continue

            # ── Step 1: Run diagnostics ──
            diag_output = execute_tool("self_diagnostics", {})

            # Short-circuit: if no issues, skip LLM call entirely
            if "No issues detected" in diag_output:
                logger.debug("Self-heal: all clear, no LLM call needed")
                continue

            # ── Step 2: LLM reasoning ──
            messages = [
                {"role": "system", "content": _SELF_HEAL_SYSTEM_PROMPT},
                {"role": "user", "content": f"Current diagnostics:\n\n{diag_output}"},
            ]

            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.post(
                        f"{OLLAMA_URL}/api/chat",
                        json={
                            "model": _SELF_HEAL_MODEL,
                            "messages": messages,
                            "stream": False,
                            "options": {"num_predict": 50},
                        },
                    )
                    if resp.status_code != 200:
                        logger.warning(f"Self-heal LLM call failed: HTTP {resp.status_code}")
                        continue
                    data = resp.json()
                    llm_response = data.get("message", {}).get("content", "").strip()
            except Exception as e:
                logger.warning("Self-heal LLM call failed [%s]", type(e).__name__)
                continue

            # ── Step 3: Parse and execute repair ──
            valid_repairs = {"restart_watcher", "clear_vram", "reindex_knowledge", "check_ollama"}

            if llm_response.startswith("REPAIR:"):
                repair_name = llm_response.split("REPAIR:", 1)[1].strip().lower()
            else:
                # Fuzzy fallback: check if any valid repair name appears in response
                repair_name = None
                for r in valid_repairs:
                    if r in llm_response.lower():
                        repair_name = r
                        break

            if repair_name and repair_name in valid_repairs:
                # Safety: don't clear VRAM during sim
                if repair_name == "clear_vram" and is_sim_running():
                    logger.info("Self-heal: blocked clear_vram during sim")
                    continue

                _log_resource_action(f"Self-heal: executing {repair_name} (LLM-decided)")
                await _emit_health_event(
                    "self_heal_action",
                    {
                        "repair": repair_name,
                        "diagnostics_summary": diag_output[:200],
                    },
                )

                # ── Step 4: Execute repair ──
                result = execute_tool("repair_subsystem", {"repair": repair_name})
                _log_resource_action(f"Self-heal result: {result[:100]}")

                # ── Step 5: Verify fix ──
                await asyncio.sleep(5)
                verify_output = execute_tool("self_diagnostics", {})
                success = "No issues detected" in verify_output
                _log_resource_action(
                    f"Self-heal verify: {'passed' if success else 'issues persist'}"
                )
                await _emit_health_event(
                    "repair_attempted",
                    {
                        "repair": repair_name,
                        "success": success,
                    },
                )

            elif "NO_ACTION" in llm_response.upper():
                logger.debug("Self-heal: LLM says no action needed")
            else:
                logger.warning(f"Self-heal: unexpected LLM response: {llm_response[:100]}")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            safe_errors.log_exception(logger, "Self-heal loop error", e)
            await asyncio.sleep(30)


# ── CIRCUIT BREAKER FOR OLLAMA ────────────────────────────────────────
# Prevents hammering Ollama when it's down, with exponential backoff.

_circuit_breaker = {
    "state": "closed",  # closed (normal), open (blocking), half-open (testing)
    "failures": 0,
    "last_failure": 0.0,
    "backoff_sec": 5,  # starts at 5s, doubles up to 60s
}
_CB_FAILURE_THRESHOLD = 3
_CB_MAX_BACKOFF = 60


def _cb_check() -> bool:
    """Returns True if calls should proceed, False if circuit is open."""
    if _circuit_breaker["state"] == "closed":
        return True
    if _circuit_breaker["state"] == "open":
        elapsed = _time.time() - _circuit_breaker["last_failure"]
        if elapsed >= _circuit_breaker["backoff_sec"]:
            _circuit_breaker["state"] = "half-open"
            return True
        return False
    return True  # half-open: allow one test call


def _cb_success():
    """Record a successful Ollama call."""
    prev_state = _circuit_breaker["state"]
    _circuit_breaker["state"] = "closed"
    _circuit_breaker["failures"] = 0
    _circuit_breaker["backoff_sec"] = 5
    if prev_state != "closed":
        _record_health_event("circuit_breaker_changed", {"state": "closed", "prev": prev_state})


def _cb_failure():
    """Record a failed Ollama call."""
    _circuit_breaker["failures"] += 1
    _circuit_breaker["last_failure"] = _time.time()
    if _circuit_breaker["failures"] >= _CB_FAILURE_THRESHOLD:
        prev_state = _circuit_breaker["state"]
        _circuit_breaker["state"] = "open"
        _circuit_breaker["backoff_sec"] = min(_circuit_breaker["backoff_sec"] * 2, _CB_MAX_BACKOFF)
        if prev_state != "open":
            _record_health_event(
                "circuit_breaker_changed",
                {
                    "state": "open",
                    "failures": _circuit_breaker["failures"],
                    "backoff_sec": _circuit_breaker["backoff_sec"],
                },
            )


# ── LIFESPAN ───────────────────────────────────────────────────────────


async def _rag_bootstrap() -> None:
    """Initial knowledge ingestion + file watcher (can take a long time)."""
    if not RAG_AVAILABLE:
        return
    try:
        results = await asyncio.to_thread(ingest_all)
        ingested = [r for r in results if r["status"] == "ok"]
        if ingested:
            logger.info(f"Initial ingestion: {len(ingested)} file(s)")
        start_watcher()
        logger.info("File watcher started")
    except Exception as e:
        safe_errors.log_exception(logger, "RAG startup failed", e)


async def _post_startup_cloud_and_models() -> None:
    """
    Cloud + Ollama model registry checks (can be slow when Ollama is down).
    Runs after the server is accepting HTTP so late-bound checks do not block startup.
    """
    t0 = _time.monotonic()
    try:
        cloud = await is_cloud_available()
        logger.info(f"Cloud available: {cloud}")
        logger.info(f"GPU utilization: {get_gpu_utilization()}%")

        primary_models = [
            ("PRIMARY", os.getenv("DELTAI_STRONG_MODEL", "deltai-qwen14b")),
            ("PRIMARY", os.getenv("DELTAI_MODEL", "deltai-qwen3b")),
        ]
        backup_models = [
            ("BACKUP", os.getenv("DELTAI_BACKUP_STRONG_MODEL", "deltai-nemo")),
            ("BACKUP", os.getenv("DELTAI_BACKUP_MODEL", "deltai-fallback")),
        ]
        for role, model in primary_models + backup_models:
            if not model.strip():
                continue
            exists = await check_model_exists(model)
            if exists:
                logger.info(f"Model check: {model} ({role}) — OK")
            elif role == "PRIMARY":
                logger.error(f"Model check: {model} ({role}) — MISSING (system degraded)")
            else:
                logger.warning(
                    f"Model check: {model} ({role}) — MISSING (emergency generator unavailable)"
                )
    finally:
        elapsed_ms = int((_time.monotonic() - t0) * 1000)
        logger.debug(f"Deferred cloud/model checks finished in {elapsed_ms}ms")


@asynccontextmanager
async def lifespan(app: FastAPI):
    t_life = _time.monotonic()
    arch_guard_task = None
    arch_guard_stop: asyncio.Event | None = None

    rag_bootstrap_task = asyncio.create_task(_rag_bootstrap()) if RAG_AVAILABLE else None

    # ── Persistence ──
    try:
        init_db()
        loaded = load_history(HISTORY_MAX_TURNS)
        _conversation_history.extend(loaded)
        logger.info(f"Loaded {len(loaded) // 2} conversation turns from DB")
        init_budget_from_db()
    except Exception as e:
        safe_errors.log_exception(logger, "Persistence startup failed", e)

    post_startup_task = asyncio.create_task(_post_startup_cloud_and_models())

    # ── Start background tasks ──
    health_task = asyncio.create_task(_backup_health_loop())
    resource_task = asyncio.create_task(_resource_manager_loop())
    self_heal_task = asyncio.create_task(_ai_self_heal_loop())
    ingest_task = asyncio.create_task(_ingest_pipeline_worker())

    if arch_guard_scheduler_loop is not None:
        arch_guard_stop = asyncio.Event()
        arch_guard_task = asyncio.create_task(arch_guard_scheduler_loop(arch_guard_stop))

    pre_yield_ms = int((_time.monotonic() - t_life) * 1000)
    logger.debug(f"Lifespan reached yield (pre-yield startup {pre_yield_ms}ms)")

    yield

    # ── Shutdown ──
    post_startup_task.cancel()
    try:
        await post_startup_task
    except asyncio.CancelledError:
        pass

    if rag_bootstrap_task:
        rag_bootstrap_task.cancel()
        try:
            await rag_bootstrap_task
        except asyncio.CancelledError:
            pass

    for task in (health_task, resource_task, self_heal_task, ingest_task):
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    if arch_guard_stop is not None:
        arch_guard_stop.set()
    if arch_guard_task is not None:
        arch_guard_task.cancel()
        try:
            await arch_guard_task
        except asyncio.CancelledError:
            pass

    if RAG_AVAILABLE:
        try:
            stop_watcher()
            logger.info("File watcher stopped")
        except Exception as e:
            safe_errors.log_exception(logger, "Watcher shutdown failed", e)

    if EXTENSIONS_AVAILABLE:
        try:
            await shutdown_extensions()
        except Exception as e:
            safe_errors.log_exception(logger, "Extensions shutdown failed", e)



def _sanitize_python_json(text: str) -> str:
    """Replace Python literals (True/False/None) with JSON equivalents."""
    text = re.sub(r"(?<=[\s:,\[])True(?=[\s,}\]])", "true", text)
    text = re.sub(r"(?<=[\s:,\[])False(?=[\s,}\]])", "false", text)
    text = re.sub(r"(?<=[\s:,\[])None(?=[\s,}\]])", "null", text)
    return text


def _fix_windows_paths(text: str) -> str:
    """Fix invalid JSON escape sequences from Windows paths (e.g., unescaped backslashes in JSON strings)."""
    # Replace single backslashes that aren't already valid JSON escapes
    # Valid JSON escapes: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)


def _safe_json_loads(text: str):
    """Parse JSON with fallbacks for Python-style literals, single quotes, and Windows paths."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try fixing Python literals
    sanitized = _sanitize_python_json(text)
    try:
        return json.loads(sanitized)
    except json.JSONDecodeError:
        pass
    # Try fixing Windows path backslashes
    fixed = _fix_windows_paths(sanitized)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    # Last resort: replace single quotes with double quotes
    sq = sanitized.replace("'", '"')
    try:
        return json.loads(sq)
    except json.JSONDecodeError:
        return json.loads(_fix_windows_paths(sq))


def _extract_balanced_braces(text: str, start: int) -> str:
    """Extract a balanced {...} substring starting at text[start] which must be '{'."""
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _extract_balanced_brackets(text: str, start: int) -> str:
    """Extract a balanced [...] substring starting at text[start] which must be '['."""
    if start >= len(text) or text[start] != "[":
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _extract_tool_from_dict(obj: dict):
    """Given a parsed dict, extract (name, params) if it looks like a tool call."""
    # Direct style: {"name": "tool", "parameters"|"arguments": {...}}
    if "name" in obj:
        params = obj.get("parameters", obj.get("arguments", {}))
        return (obj["name"], params if isinstance(params, dict) else {})
    # Ollama-style: {"function": {"name": ..., "arguments": ...}}
    if "function" in obj and isinstance(obj["function"], dict):
        fn = obj["function"]
        if "name" in fn:
            params = fn.get("arguments", fn.get("parameters", {}))
            return (fn["name"], params if isinstance(params, dict) else {})
    return None


def try_parse_text_tool_call(content: str):
    """Parse tool calls from model text output. Returns (name, params) or None.

    Handles:
    - {"name":"tool","parameters":{...}} or "arguments" variant
    - ```json code blocks containing tool calls
    - Ollama-style tool_calls arrays: [{"function":{"name":"...","arguments":{...}}}]
    - JSON embedded in surrounding explanatory text (balanced-brace extraction)
    - Nested brace objects in parameters
    - Multiple tool calls in text (takes first only)
    """
    if not content:
        return None
    text = content.strip()

    # 1) Markdown code block — highest confidence signal
    md_match = re.search(r"```(?:json)?\s*([\[{][\s\S]+?[}\]])\s*```", text, re.DOTALL)
    if md_match:
        try:
            obj = _safe_json_loads(md_match.group(1))
            if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
                result = _extract_tool_from_dict(obj[0])
                if result:
                    return result
            elif isinstance(obj, dict):
                result = _extract_tool_from_dict(obj)
                if result:
                    return result
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

    # 2) Exact full-text JSON parse (no surrounding text)
    try:
        obj = _safe_json_loads(text)
        if isinstance(obj, dict):
            result = _extract_tool_from_dict(obj)
            if result:
                return result
        if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
            result = _extract_tool_from_dict(obj[0])
            if result:
                return result
    except (json.JSONDecodeError, ValueError):
        pass

    # 3) Extract JSON from surrounding text using balanced brace/bracket matching.
    #    Scans left-to-right, returns first valid tool call (prevents infinite loops
    #    when multiple calls are present).
    for i, ch in enumerate(text):
        if ch == "{":
            blob = _extract_balanced_braces(text, i)
            if blob and len(blob) > 10:
                try:
                    obj = _safe_json_loads(blob)
                    if isinstance(obj, dict):
                        result = _extract_tool_from_dict(obj)
                        if result:
                            return result
                except (json.JSONDecodeError, ValueError):
                    pass
        elif ch == "[":
            blob = _extract_balanced_brackets(text, i)
            if blob and len(blob) > 10:
                try:
                    obj = _safe_json_loads(blob)
                    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
                        result = _extract_tool_from_dict(obj[0])
                        if result:
                            return result
                except (json.JSONDecodeError, ValueError):
                    pass

    return None


# ── GREETING SHORT-CIRCUIT ─────────────────────────────────────────────
# Intercepts obvious simple messages BEFORE they hit any model.
# Fixes 3B over-triggering tools on greetings — model-agnostic, zero latency.

_GREETING_MAP = {
    # Greetings — warm but brief
    "hey": "Hey. What's going on?",
    "hi": "Hey. What's going on?",
    "hello": "Hey, Ethan. What do you need?",
    "hola": "Hey, Ethan. What do you need?",
    "oh": "Hey. What do you need?",  # common Whisper mis-transcription of "hello"
    "yo": "Yo. What's up?",
    "hey deltai": "Hey. I'm here.",
    "hi deltai": "Hey. Ready when you are.",
    "hello deltai": "Hey, Ethan. What are we working on?",
    "hey e3": "Hey. I'm here.",
    "hi e3": "Hey. What do you need?",
    # What's up
    "sup": "All green on my end. What do you need?",
    "whats up": "All green on my end. What's going on?",
    "what's up": "All green on my end. What do you need?",
    "wassup": "All systems nominal. What's up?",
    # Morning / evening
    "morning": "Morning, Ethan. Ready when you are.",
    "good morning": "Morning. Systems are all green. What's the plan?",
    "morning deltai": "Morning, Ethan. Ready when you are.",
    "morning e3": "Morning. What are we working on?",
    "gm": "Morning, Ethan. Let's get it.",
    "evening": "Evening, Ethan. What do you need?",
    "good evening": "Evening. Everything's running smooth. What's up?",
    # Night / goodbye
    "night": "Night, Ethan. I'll keep watch.",
    "goodnight": "Night, Ethan. I'll keep watch.",
    "good night": "Get some rest. I'll be here.",
    "gn": "Night. I'll be here when you're back.",
    "bye": "Copy. See you later.",
    "later": "Later. I'll be here.",
    "peace": "Take it easy, Ethan.",
    "see ya": "See you. I'll hold down the fort.",
    "goodbye": "Later, Ethan. I'll be here.",
    # Status checks
    "you there?": "I'm here. What do you need?",
    "you there": "Right here. What's up?",
    "online?": "Online and ready. What do you need?",
    "awake?": "Always. What's going on?",
    "you up?": "Always on. What do you need?",
    "deltai?": "I'm here. What's up?",
    "e3?": "Right here.",
    # Thanks
    "thanks": "That's what I'm here for.",
    "thank you": "Anytime, Ethan.",
    "thanks deltai": "That's what I'm here for.",
    "thx": "No problem.",
}


def _check_greeting(message: str) -> str | None:
    """
    Check if the message is a simple greeting/farewell.
    Returns a canned deltai response or None if not a greeting.
    """
    text = message.strip().lower().rstrip("!.,?")
    return _GREETING_MAP.get(text)


# ── RAG CONTEXT BUILDER ────────────────────────────────────────────────


def build_rag_context(
    user_message: str, source_filter: str = None, max_age_sec: float = None
) -> str:
    if not RAG_AVAILABLE:
        return ""
    try:
        from memory import generate_sub_queries, iterative_query_knowledge

        # Round 1: Standard retrieval
        matches = query_knowledge(
            user_message,
            n_results=3,
            threshold=0.75,
            source_filter=source_filter,
            max_age_sec=max_age_sec,
        )
        # Round 2: Iterative refinement if first round has results
        if matches and len(matches) >= 1:
            sub_queries = generate_sub_queries(user_message, matches)
            if sub_queries:
                matches = iterative_query_knowledge(
                    user_message,
                    sub_queries=sub_queries,
                    n_results=5,
                    threshold=0.75,
                    source_filter=source_filter,
                    max_age_sec=max_age_sec,
                )
        if not matches:
            return ""
        context_parts = ["[KNOWLEDGE CONTEXT — from your ingested documents]"]
        for m in matches:
            context_parts.append(
                f"[Source: {m['source']} | Relevance: {1 - m['distance']:.0%}]\n{m['text']}"
            )
        context_parts.append("[END CONTEXT]\n")
        return "\n\n".join(context_parts)
    except Exception as e:
        safe_errors.log_exception(logger, "RAG query failed", e)
        return ""


# ── CHAT (with smart routing) ──────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str
    deep: bool = False
    force_local: bool = False


MAX_TOOL_ROUNDS = 6

# ── STREAMING INGEST PIPELINE ─────────────────────────────────────────
# Async queue-based ingest: /ingest returns immediately, background worker
# batches and processes embeddings efficiently.

_INGEST_QUEUE_MAX = int(os.getenv("INGEST_QUEUE_MAX", "500"))
_INGEST_FLUSH_INTERVAL = float(os.getenv("INGEST_FLUSH_INTERVAL", "2.0"))  # seconds
_INGEST_FLUSH_BATCH_SIZE = int(os.getenv("INGEST_FLUSH_BATCH_SIZE", "10"))

_ingest_queue: asyncio.Queue | None = None
_ingest_metrics = {
    "queued": 0,
    "processed": 0,
    "errors": 0,
    "avg_latency_ms": 0.0,
    "last_flush": 0.0,
}


async def _ingest_pipeline_worker():
    """
    Background worker: drain ingest queue, batch-embed, write to ChromaDB.
    Flushes every FLUSH_INTERVAL seconds or when FLUSH_BATCH_SIZE items accumulate.
    """
    global _ingest_queue
    _ingest_queue = asyncio.Queue(maxsize=_INGEST_QUEUE_MAX)
    await asyncio.sleep(5)  # let startup finish
    logger.info("Ingest pipeline worker started")

    while True:
        batch = []
        try:
            # Collect items up to batch size or flush interval
            deadline = _time.time() + _INGEST_FLUSH_INTERVAL
            while len(batch) < _INGEST_FLUSH_BATCH_SIZE:
                timeout = max(0.1, deadline - _time.time())
                try:
                    item = await asyncio.wait_for(_ingest_queue.get(), timeout=timeout)
                    batch.append(item)
                except TimeoutError:
                    break

            if not batch or not RAG_AVAILABLE:
                continue

            # Process batch
            start = _time.time()
            items_for_batch = [
                {
                    "source": it["source"],
                    "context": it["context"],
                    "ttl": it["ttl"],
                    "tags": it["tags"],
                }
                for it in batch
            ]
            try:
                await asyncio.to_thread(ingest_context_batch, items_for_batch)
                _ingest_metrics["processed"] += len(batch)
                elapsed = (_time.time() - start) * 1000
                _ingest_metrics["avg_latency_ms"] = (
                    _ingest_metrics["avg_latency_ms"] * 0.8 + elapsed * 0.2
                )
            except Exception as e:
                _ingest_metrics["errors"] += len(batch)
                logger.warning("Ingest pipeline batch failed [%s]", type(e).__name__)

            _ingest_metrics["last_flush"] = _time.time()

            # Forward alerts
            for it in batch:
                if "alert" in it.get("tags", []):
                    alert = {
                        "source": it["source"],
                        "context": it["context"],
                        "tags": it["tags"],
                        "ts": _time.time(),
                        "priority": "critical"
                        if "critical" in it["tags"]
                        else "high"
                        if "high" in it["tags"]
                        else "normal",
                    }
                    _recent_alerts.append(alert)
                    await _broadcast_alert(alert)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            safe_errors.log_exception(logger, "Ingest pipeline error", e)
            await asyncio.sleep(1)

