from contextlib import asynccontextmanager
from collections import deque
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse, PlainTextResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import httpx
import json
import re
import os
import time as _time
import asyncio
import base64
import psutil
import platform
import logging
from dotenv import load_dotenv

import safe_errors

from tools.definitions import TOOLS, filter_tools
from tools.executor import execute_tool
from router import (route, is_cloud_available, is_cloud_available_sync,
                    get_gpu_utilization, get_vram_free_mb, classify_complexity,
                    is_sim_running, get_budget_status, record_cloud_usage,
                    _pick_local_model, get_backup_model, check_model_health,
                    check_model_exists, init_budget_from_db)
from anthropic_client import stream_chat as anthropic_stream
from persistence import (init_db, load_history, save_history_pair,
                         clear_history as db_clear_history, trim_history,
                         save_reasoning_trace, find_similar_traces, prune_old_traces,
                         _serialize_embedding,
                         save_quality_score, save_routing_feedback,
                         save_knowledge_gap, get_unresolved_gaps, resolve_knowledge_gap,
                         get_routing_stats)

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
    from memory import (query_knowledge, ingest_all, get_memory_stats, get_file_details,
                        remove_file, ingest_context, cleanup_expired, ingest_context_batch,
                        compact_warm_to_cold, get_cold_stats,
                        KNOWLEDGE_PATH)
    from watcher import start_watcher, stop_watcher, watcher_running
    RAG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RAG system unavailable: {e}")

# ── TRAINING IMPORTS ───────────────────────────────────────────────────
TRAINING_AVAILABLE = False
try:
    from training import (
        list_datasets, create_dataset, delete_dataset,
        get_dataset, add_example, remove_example,
        export_dataset, get_training_status,
        start_training, stop_training, auto_capture,
    )
    TRAINING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Training system unavailable: {e}")


# ── VOICE IMPORTS ─────────────────────────────────────────────────────
VOICE_AVAILABLE = False
try:
    from voice import (transcribe_audio, synthesize_speech, get_voice_status,
                       record_audio, VOICE_ENABLED)
    VOICE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Voice system unavailable: {e}")

# ── EXTENSIONS IMPORTS ─────────────────────────────────────────────────
EXTENSIONS_AVAILABLE = False
try:
    from extensions import load_extensions, get_extension_tools, shutdown_extensions
    from tools.definitions import _merge_extension_tools
    EXTENSIONS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Extensions system unavailable: {e}")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DELTAI_MODEL  = os.getenv("DELTAI_MODEL", "deltai")
BACKUP_MAX_RETRIES = int(os.getenv("BACKUP_MAX_RETRIES", "2"))
TELEMETRY_API_URL = os.getenv("TELEMETRY_API_URL", "").strip()

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── EMERGENCY BACKUP STATE ────────────────────────────────────────────

_backup_health_status: dict = {}   # {model: {"healthy": bool, "last_check": float}}
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


def _append_to_history(user_message: str, assistant_response: str,
                       chat_metadata: dict = None):
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
        logger.warning(f"Failed to persist history: {e}")

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
        logger.debug(f"Quality scoring failed: {e}")

    # ── Routing feedback ──
    if metadata.get("model") and quality_result:
        try:
            import hashlib
            qhash = hashlib.md5(user_message.strip().lower().encode()).hexdigest()
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
            logger.debug(f"Routing feedback save failed: {e}")

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
                "deltai-auto", user_message, assistant_response,
                quality_score=quality_result["score"] if quality_result else 0.5,
                metadata=metadata,
            )
        except Exception:
            # Fall back to basic auto_capture
            try:
                auto_capture("deltai-auto", user_message, assistant_response)
            except Exception:
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
    sentences = _re.split(r'[.!?]\s+', content)
    key_sentences = []
    for s in sentences:
        s = s.strip()
        if not s or len(s) < 10:
            continue
        # Keep sentences with numbers, units, or technical terms
        has_numbers = bool(_re.search(r'\d+\.?\d*\s*(?:MB|GB|ms|s|°C|%|rpm|kg|N|Pa)', s))
        has_code = '`' in s or '```' in s
        has_conclusion = any(w in s.lower() for w in [
            "result", "therefore", "conclusion", "answer", "solution",
            "should", "recommend", "optimal", "best",
        ])
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
    first_sentence = content.split('.')[0].split('?')[0].split('!')[0]
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
                result.append({"role": "assistant", "content": _compress_turn(assistant_msg["content"])})
        else:
            # Old: summarize
            result.append({"role": "user", "content": _summarize_turn(user_msg["content"])})
            if assistant_msg:
                result.append({"role": "assistant", "content": _summarize_turn(assistant_msg["content"])})

    return result


async def _try_ollama_inference(client: httpx.AsyncClient, model: str,
                                messages: list, tools: list | None = None,
                                num_gpu: int | None = None,
                                num_ctx: int | None = None) -> tuple[dict | None, str | None]:
    """
    Attempt a single Ollama inference call with circuit breaker protection.
    Returns (data, None) on success or (None, error_string) on failure.
    Supports num_gpu (partial GPU offloading) and num_ctx (dynamic context window).
    """
    # Circuit breaker: don't hammer Ollama when it's down
    if not _cb_check():
        return None, f"Circuit breaker open — Ollama unreachable (backoff {_circuit_breaker['backoff_sec']}s)"

    payload = {"model": model, "messages": messages, "stream": False}
    if tools:
        payload["tools"] = tools
    # Dynamic GPU layer offloading + context window sizing
    options = {}
    if num_gpu is not None:
        options["num_gpu"] = num_gpu
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
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
    client: httpx.AsyncClient, model: str, messages: list,
    tools: list | None, backup_model: str | None,
    num_gpu: int | None = None, num_ctx: int | None = None,
) -> tuple[dict | None, str | None, str, bool]:
    """
    Try primary model with retries, then emergency backup as last resort.

    Returns (data, error, used_model, is_emergency).
    """
    # ── Attempt primary (with retries) ──
    last_error = None
    for attempt in range(1 + BACKUP_MAX_RETRIES):
        data, err = await _try_ollama_inference(client, model, messages, tools,
                                                 num_gpu=num_gpu, num_ctx=num_ctx)
        if data is not None:
            return data, None, model, False
        last_error = err
        if attempt < BACKUP_MAX_RETRIES:
            logger.warning(f"Primary model {model} failed (attempt {attempt + 1}): {err} — retrying in 3s")
            await asyncio.sleep(3)

    # ── Primary exhausted — engage emergency backup ──
    # Backups run with default settings (no partial offload)
    if backup_model:
        logger.error(f"PRIMARY MODEL DOWN: {model} failed {1 + BACKUP_MAX_RETRIES} attempts. "
                     f"Engaging emergency backup: {backup_model}")
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

_REACT_SYSTEM_PROMPT = """You are in structured reasoning mode. For each step:

THINK: State what you need to figure out or what information you're missing.
ACT: Call a tool to gather information, calculate, or look something up. If no tool is needed, write ACT: none.
OBSERVE: Analyze the result you got back.
CONFIDENCE: Rate your confidence: HIGH (>80%), MEDIUM (50-80%), LOW (<50%).

After sufficient information is gathered (max {max_iter} iterations), provide your final answer.

Format your response EXACTLY as:
THINK: [your reasoning]
ACT: [tool call or "none"]
OBSERVE: [analysis of results]
CONFIDENCE: [HIGH/MEDIUM/LOW]

Confidence protocol:
- HIGH: You have enough information. Proceed to FINAL.
- MEDIUM after 2+ iterations: Provide your best answer with [CONFIDENCE: MEDIUM] and note what additional info would help.
- LOW: Use ACT to gather more data, or write CLARIFY: [question for the operator] if you need input.

When ready to answer:
FINAL: [your complete answer]"""


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


async def _react_reasoning_loop(client, model: str, user_message: str,
                                 rag_context: str, history: list,
                                 tools: list, execute_fn,
                                 num_gpu: int | None = None,
                                 num_ctx: int | None = None) -> tuple[str, list]:
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
            logger.debug(f"Trace retrieval failed: {e}")

    react_system = _REACT_SYSTEM_PROMPT.format(max_iter=_REACT_MAX_ITERATIONS)
    context_prefix = f"{rag_context}\n" if rag_context else ""

    messages = history + [
        {"role": "system", "content": react_system},
        {"role": "user", "content": f"{prior_context}{context_prefix}{user_message}"},
    ]

    for iteration in range(_REACT_MAX_ITERATIONS):
        data, err = await _try_ollama_inference(
            client, model, messages, tools,
            num_gpu=num_gpu, num_ctx=num_ctx
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
            _save_react_trace(user_message, "", content, events, confidence=confidence, success=False)
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
            messages.append({"role": "user",
                           "content": "Continue reasoning. If you have enough information, respond with FINAL: followed by your answer."})

    # Max iterations reached — extract best answer from last response
    if content:
        # Try to find any FINAL or just use the last content
        if "FINAL:" in content:
            final_text = content.split("FINAL:", 1)[1].strip()
            _save_react_trace(user_message, final_text, content, events)
            return final_text, events
        _save_react_trace(user_message, content, content, events)
        return content, events
    _save_react_trace(user_message, "Reasoning loop completed without a clear answer.", content or "", events)
    return "Reasoning loop completed without a clear answer.", events


def _save_react_trace(user_message: str, final_text: str, content: str, events: list,
                       confidence: str = "unknown", success: bool = None):
    """Save a reasoning trace after a ReAct loop completes."""
    if os.getenv("REASONING_TRACE_ENABLED", "true").lower() not in ("true", "1", "yes"):
        return
    try:
        from memory import get_embeddings
        query_emb = get_embeddings([user_message])[0]
        query_emb_bytes = _serialize_embedding(query_emb)
        tool_seq = ",".join(ev.get("n", "") for ev in events if ev.get("t") == "tool")
        was_success = success if success is not None else ("FINAL:" in (content or "") or bool(final_text))
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
        logger.debug(f"Trace save failed: {e}")


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
    "last_vram_action": 0.0,     # timestamp of last VRAM management action
    "last_recovery": 0.0,        # timestamp of last auto-recovery attempt
    "vram_warnings": 0,          # consecutive low-VRAM readings
    "ollama_failures": 0,        # consecutive Ollama connectivity failures
    "watcher_restarts": 0,       # number of watcher auto-restarts this session
    "actions_taken": [],         # log of automatic actions (last 50)
    "last_sim_state": False,     # previous sim running state (for transition detection)
    "sim_stop_detected_at": 0.0, # when sim stop was first detected
    "pending_14b_preload": False, # waiting to preload 14B after sim stop
    "vram_history": [],          # sliding window of (timestamp, vram_free_mb) for prediction
    "priority_lowered": False,   # whether OS process priority is currently lowered
}
_RESOURCE_CHECK_INTERVAL = 30    # seconds between resource checks
_VRAM_CRITICAL_MB = 1500        # below this, aggressively free VRAM
_VRAM_WARN_MB = 3000            # below this, start monitoring closely
_MAX_WATCHER_RESTARTS = 5       # don't restart watcher more than this
_OLLAMA_FAILURE_THRESHOLD = 3   # consecutive failures before auto-restart attempt
_RESOURCE_ACTION_COOLDOWN = 60  # seconds between automatic VRAM actions
_VRAM_HISTORY_WINDOW = 60       # seconds of VRAM history to keep for prediction
_VRAM_DECLINE_RATE_THRESHOLD = 100  # MB/s — preemptive unload if declining faster
_GPU_TEMP_THROTTLE = 80         # celsius — switch to smaller model above this


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
        import subprocess as _sp
        target = psutil.BELOW_NORMAL_PRIORITY_CLASS if lower else psutil.NORMAL_PRIORITY_CLASS
        label = "BELOW_NORMAL" if lower else "NORMAL"

        # Adjust own process
        own = psutil.Process()
        own.nice(target)

        # Adjust Ollama process(es)
        for proc in psutil.process_iter(['name', 'pid']):
            try:
                if proc.info['name'] and 'ollama' in proc.info['name'].lower():
                    proc.nice(target)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        _resource_state["priority_lowered"] = lower
        _log_resource_action(f"Process priority → {label}")
    except Exception as e:
        logger.debug(f"Priority adjustment failed: {e}")


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
                                        await client.post(f"{OLLAMA_URL}/api/generate",
                                            json={"model": m["name"], "prompt": "", "keep_alive": 0},
                                            timeout=10)
                                        _log_resource_action(f"Unloaded {m['name']} (GPU temp {gpu_temp}°C)")
                                        _resource_state["last_vram_action"] = now
                    except Exception:
                        pass

            # ── 0b. PREDICTIVE VRAM — preemptive action on rapid decline ──
            decline_rate = _predict_vram_decline()
            if (decline_rate > _VRAM_DECLINE_RATE_THRESHOLD and
                    vram_free < _VRAM_WARN_MB * 1.5 and
                    now - _resource_state["last_vram_action"] > _RESOURCE_ACTION_COOLDOWN):
                try:
                    async with httpx.AsyncClient(timeout=5) as client:
                        resp = await client.get(f"{OLLAMA_URL}/api/ps")
                        if resp.status_code == 200:
                            for m in resp.json().get("models", []):
                                if "14b" in m.get("name", "").lower():
                                    await client.post(f"{OLLAMA_URL}/api/generate",
                                        json={"model": m["name"], "prompt": "", "keep_alive": 0},
                                        timeout=10)
                                    _log_resource_action(
                                        f"Preemptive unload {m['name']} "
                                        f"(VRAM declining {decline_rate:.0f}MB/s, {vram_free}MB free)")
                                    _resource_state["last_vram_action"] = now
                except Exception:
                    pass

            # ── 1. VRAM MANAGEMENT ──
            if vram_free < _VRAM_CRITICAL_MB:
                _resource_state["vram_warnings"] += 1
                # Critical VRAM — aggressive action after 2 consecutive readings
                if (_resource_state["vram_warnings"] >= 2 and
                        now - _resource_state["last_vram_action"] > _RESOURCE_ACTION_COOLDOWN):
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
                                        await client.post(f"{OLLAMA_URL}/api/generate",
                                            json={"model": m["name"], "prompt": "", "keep_alive": 0},
                                            timeout=10)
                                        _log_resource_action(f"Unloaded {m['name']} (VRAM critical: {vram_free}MB, sim active)")
                                    elif vram_free < 1000 and len(loaded) > 1:
                                        # Extreme pressure — unload everything except smallest
                                        sizes = [(m2.get("name", ""), m2.get("size_vram", m2.get("size", 0)))
                                                 for m2 in loaded]
                                        sizes.sort(key=lambda x: x[1], reverse=True)
                                        for model_name, _ in sizes[:-1]:  # keep smallest
                                            await client.post(f"{OLLAMA_URL}/api/generate",
                                                json={"model": model_name, "prompt": "", "keep_alive": 0},
                                                timeout=10)
                                            _log_resource_action(f"Unloaded {model_name} (VRAM extreme: {vram_free}MB)")
                                        break
                                _resource_state["last_vram_action"] = now
                    except Exception as e:
                        logger.debug(f"VRAM management failed: {e}")

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
                                    await mc.post(f"{OLLAMA_URL}/api/generate",
                                        json={"model": m["name"], "prompt": "", "keep_alive": 0},
                                        timeout=10)
                                    _log_resource_action(f"Unloaded {m['name']} (sim started)")
                                    await _emit_health_event("model_unloaded", {"model": m["name"], "reason": "sim_start"})
                        # Preload 3B if VRAM allows
                        vram_now = get_vram_free_mb()
                        sim_model = os.getenv("DELTAI_SIM_MODEL", os.getenv("DELTAI_MODEL", "deltai-qwen3b"))
                        if vram_now >= 2500:
                            await mc.post(f"{OLLAMA_URL}/api/generate",
                                json={"model": sim_model, "prompt": "ping", "keep_alive": "5m",
                                      "options": {"num_predict": 1}}, timeout=60)
                            _log_resource_action(f"Preloaded {sim_model} (sim started, {vram_now}MB free)")
                            await _emit_health_event("model_loaded", {"model": sim_model, "reason": "sim_start"})
                except Exception as e:
                    logger.debug(f"Sim-start model swap failed: {e}")
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
                                await mc.post(f"{OLLAMA_URL}/api/generate",
                                    json={"model": strong_model, "prompt": "ping",
                                          "keep_alive": "5m", "options": {"num_predict": 1}},
                                    timeout=60)
                                _log_resource_action(f"Preloaded {strong_model} (sim stopped, {vram_now}MB free)")
                                await _emit_health_event("model_loaded", {"model": strong_model, "reason": "sim_stop"})
                        except Exception as e:
                            logger.debug(f"Post-sim 14B preload failed: {e}")
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
                    _log_resource_action(f"Ollama unreachable ({_resource_state['ollama_failures']} failures), attempting restart")
                    try:
                        import subprocess
                        subprocess.Popen(
                            ["ollama", "serve"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        )
                        _resource_state["last_recovery"] = now
                        _resource_state["ollama_failures"] = 0
                        await asyncio.sleep(5)  # give it time to start
                    except Exception as e:
                        logger.error(f"Ollama auto-restart failed: {e}")

            # ── 4. WATCHER RECOVERY ──
            if RAG_AVAILABLE:
                try:
                    if not watcher_running() and _resource_state["watcher_restarts"] < _MAX_WATCHER_RESTARTS:
                        _log_resource_action("File watcher died, restarting")
                        stop_watcher()
                        start_watcher()
                        _resource_state["watcher_restarts"] += 1
                except Exception as e:
                    logger.debug(f"Watcher recovery check failed: {e}")

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
                        max_age_days=30,
                        max_count=int(os.getenv("REASONING_TRACE_MAX", "500"))
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
            logger.error(f"Resource manager error: {e}")
            await asyncio.sleep(10)


# ── AI-DRIVEN SELF-HEAL LOOP ──────────────────────────────────────────
# Periodically runs diagnostics, has the LLM reason about issues, and
# executes repairs autonomously. Only runs when idle + not in session.

_SELF_HEAL_ENABLED = os.getenv("SELF_HEAL_ENABLED", "true").lower() in ("true", "1", "yes")
_SELF_HEAL_INTERVAL = int(os.getenv("SELF_HEAL_INTERVAL_SEC", "300"))  # 5 minutes
_SELF_HEAL_MODEL = os.getenv("DELTAI_MODEL", "deltai-qwen3b")

_SELF_HEAL_SYSTEM_PROMPT = """You are deltai's internal diagnostics AI. You receive a system diagnostics report and must decide if any repairs are needed.

RULES:
- Only suggest repairs if there are actual issues (DOWN, ERROR, STOPPED, WARNING, DEGRADED)
- Available repairs: restart_watcher, clear_vram, reindex_knowledge, check_ollama
- If no issues found, respond with exactly: NO_ACTION
- If repair needed, respond with exactly: REPAIR:<repair_name>
- Only suggest ONE repair per cycle (most critical first)
- Never suggest clear_vram during an active GPU focus session
- Never suggest repairs for cosmetic warnings

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
                    resp = await client.post(f"{OLLAMA_URL}/api/chat", json={
                        "model": _SELF_HEAL_MODEL,
                        "messages": messages,
                        "stream": False,
                        "options": {"num_predict": 50},
                    })
                    if resp.status_code != 200:
                        logger.warning(f"Self-heal LLM call failed: HTTP {resp.status_code}")
                        continue
                    data = resp.json()
                    llm_response = data.get("message", {}).get("content", "").strip()
            except Exception as e:
                logger.warning(f"Self-heal LLM call failed: {e}")
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
                await _emit_health_event("self_heal_action", {
                    "repair": repair_name,
                    "diagnostics_summary": diag_output[:200],
                })

                # ── Step 4: Execute repair ──
                result = execute_tool("repair_subsystem", {"repair": repair_name})
                _log_resource_action(f"Self-heal result: {result[:100]}")

                # ── Step 5: Verify fix ──
                await asyncio.sleep(5)
                verify_output = execute_tool("self_diagnostics", {})
                success = "No issues detected" in verify_output
                _log_resource_action(f"Self-heal verify: {'passed' if success else 'issues persist'}")
                await _emit_health_event("repair_attempted", {
                    "repair": repair_name, "success": success,
                })

            elif "NO_ACTION" in llm_response.upper():
                logger.debug("Self-heal: LLM says no action needed")
            else:
                logger.warning(f"Self-heal: unexpected LLM response: {llm_response[:100]}")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Self-heal loop error: {e}")
            await asyncio.sleep(30)


# ── CIRCUIT BREAKER FOR OLLAMA ────────────────────────────────────────
# Prevents hammering Ollama when it's down, with exponential backoff.

_circuit_breaker = {
    "state": "closed",    # closed (normal), open (blocking), half-open (testing)
    "failures": 0,
    "last_failure": 0.0,
    "backoff_sec": 5,     # starts at 5s, doubles up to 60s
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
        _circuit_breaker["backoff_sec"] = min(
            _circuit_breaker["backoff_sec"] * 2, _CB_MAX_BACKOFF
        )
        if prev_state != "open":
            _record_health_event("circuit_breaker_changed", {
                "state": "open", "failures": _circuit_breaker["failures"],
                "backoff_sec": _circuit_breaker["backoff_sec"],
            })


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
        logger.error(f"RAG startup failed: {e}")


async def _post_startup_cloud_and_models() -> None:
    """
    Cloud + Ollama model registry checks (can be slow when Ollama is down).
    Runs after the server is accepting HTTP so Electron waitForBackend succeeds.
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
                logger.warning(f"Model check: {model} ({role}) — MISSING (emergency generator unavailable)")
    finally:
        elapsed_ms = int((_time.monotonic() - t0) * 1000)
        logger.debug(f"Deferred cloud/model checks finished in {elapsed_ms}ms")


@asynccontextmanager
async def lifespan(app: FastAPI):
    t_life = _time.monotonic()

    rag_bootstrap_task = asyncio.create_task(_rag_bootstrap()) if RAG_AVAILABLE else None

    # ── Persistence ──
    try:
        init_db()
        loaded = load_history(HISTORY_MAX_TURNS)
        _conversation_history.extend(loaded)
        logger.info(f"Loaded {len(loaded) // 2} conversation turns from DB")
        init_budget_from_db()
    except Exception as e:
        logger.error(f"Persistence startup failed: {e}")

    post_startup_task = asyncio.create_task(_post_startup_cloud_and_models())

    # ── Start background tasks ──
    health_task = asyncio.create_task(_backup_health_loop())
    resource_task = asyncio.create_task(_resource_manager_loop())
    self_heal_task = asyncio.create_task(_ai_self_heal_loop())
    ingest_task = asyncio.create_task(_ingest_pipeline_worker())

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

    if RAG_AVAILABLE:
        try:
            stop_watcher()
            logger.info("File watcher stopped")
        except Exception as e:
            logger.error(f"Watcher shutdown failed: {e}")

    if EXTENSIONS_AVAILABLE:
        try:
            await shutdown_extensions()
        except Exception as e:
            logger.error(f"Extensions shutdown failed: {e}")

app = FastAPI(title="deltai", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=os.path.join(_HERE, "static")), name="static")

# ── EXTENSIONS ──────────────────────────────────────────────────────────
if EXTENSIONS_AVAILABLE:
    try:
        load_extensions(app)
        _merge_extension_tools(get_extension_tools())
    except Exception as _ext_err:
        logger.warning(f"Extensions failed to initialise: {_ext_err}")


# ── TEXT-AS-TOOL FALLBACK PARSER ────────────────────────────────────────

def _sanitize_python_json(text: str) -> str:
    """Replace Python literals (True/False/None) with JSON equivalents."""
    text = re.sub(r'(?<=[\s:,\[])True(?=[\s,}\]])', 'true', text)
    text = re.sub(r'(?<=[\s:,\[])False(?=[\s,}\]])', 'false', text)
    text = re.sub(r'(?<=[\s:,\[])None(?=[\s,}\]])', 'null', text)
    return text

def _fix_windows_paths(text: str) -> str:
    """Fix invalid JSON escape sequences from Windows paths (e.g., unescaped backslashes in JSON strings)."""
    # Replace single backslashes that aren't already valid JSON escapes
    # Valid JSON escapes: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
    return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)

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
    if start >= len(text) or text[start] != '{':
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None

def _extract_balanced_brackets(text: str, start: int) -> str:
    """Extract a balanced [...] substring starting at text[start] which must be '['."""
    if start >= len(text) or text[start] != '[':
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
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
    md_match = re.search(r'```(?:json)?\s*([\[{][\s\S]+?[}\]])\s*```', text, re.DOTALL)
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
        if ch == '{':
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
        elif ch == '[':
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

def build_rag_context(user_message: str, source_filter: str = None,
                      max_age_sec: float = None) -> str:
    if not RAG_AVAILABLE:
        return ""
    try:
        from memory import iterative_query_knowledge, generate_sub_queries
        # Round 1: Standard retrieval
        matches = query_knowledge(user_message, n_results=3, threshold=0.75,
                                  source_filter=source_filter, max_age_sec=max_age_sec)
        # Round 2: Iterative refinement if first round has results
        if matches and len(matches) >= 1:
            sub_queries = generate_sub_queries(user_message, matches)
            if sub_queries:
                matches = iterative_query_knowledge(
                    user_message, sub_queries=sub_queries, n_results=5,
                    threshold=0.75, source_filter=source_filter, max_age_sec=max_age_sec)
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
        logger.error(f"RAG query failed: {e}")
        return ""


# ── CHAT (with smart routing) ──────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    deep: bool = False
    force_local: bool = False
    voice_input: bool = False  # True when input came from speech — helps model correct STT errors

MAX_TOOL_ROUNDS = 6

@app.get("/")
def root():
    return FileResponse(os.path.join(_HERE, "static", "index.html"))

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Chat endpoint with smart routing + tool calling + RAG.

    Stream protocol (JSON lines):
      {"t":"route","backend":"...","model":"...","tier":N,"reason":"..."}
      {"t":"rag","n":N}
      {"t":"tool","n":"tool_name","a":{args}}
      {"t":"result","n":"tool_name","s":"summary"}
      {"t":"text","c":"chunk"}
      {"t":"done"}
      {"t":"error","c":"message"}
    """
    global _last_chat_start
    _last_chat_start = _time.time()
    # ── ROUTE DECISION ──────────────────────────────────────
    decision = await route(
        message=req.message,
        force_cloud=req.deep,
        force_local=req.force_local,
    )
    logger.info(f"Route: {decision}")

    # ── GREETING SHORT-CIRCUIT ───────────────────────────────
    greeting_response = _check_greeting(req.message)
    if greeting_response:
        async def stream_greeting():
            route_data = decision.to_dict()
            route_data["reason"] = "greeting — short-circuit"
            yield json.dumps({"t": "route", **route_data}) + "\n"
            yield json.dumps({"t": "text", "c": greeting_response}) + "\n"
            yield json.dumps({"t": "done"}) + "\n"
        logger.info(
            "Greeting short-circuit (message_len=%s reply_len=%s)",
            len(req.message or ""),
            len(greeting_response or ""),
        )
        return StreamingResponse(stream_greeting(), media_type="application/x-ndjson")

    # ── RAG CONTEXT ─────────────────────────────────────────
    rag_context = build_rag_context(req.message)

    # ── SPLIT WORKLOAD PATH ─────────────────────────────────
    # Phase 1: Local 3B model gathers data via tools (free, fast)
    # Phase 2: Cloud model reasons over the enriched context (premium quality)
    if decision.backend == "anthropic" and decision.split:
        async def stream_split():
            # ── Route event with split flag ──
            route_data = decision.to_dict()
            route_data["split"] = True
            yield json.dumps({"t": "route", **route_data}) + "\n"

            if rag_context:
                chunk_count = rag_context.count("[Source:")
                yield json.dumps({"t": "rag", "n": chunk_count}) + "\n"

            split_message = req.message
            if req.voice_input:
                split_message = f"[Voice input — may contain transcription errors. Interpret the intended meaning.]\n{req.message}"

            # ── Phase 1: Local tool gathering ──
            yield json.dumps({"t": "split_phase", "phase": 1, "c": "Gathering data locally..."}) + "\n"

            local_model, cpu_only, backup = _pick_local_model()
            if not local_model:
                # No local model — fall back to standard cloud with tools
                yield json.dumps({"t": "split_phase", "phase": 0, "c": "No local model — sending to cloud..."}) + "\n"
                full_response = ""
                async for line in anthropic_stream(
                    message=req.message, model=decision.model,
                    rag_context=rag_context, tools=TOOLS, execute_tool_fn=execute_tool,
                    history=_get_history(),
                ):
                    if line.strip().startswith('{"t":"_usage"'):
                        try:
                            usage = json.loads(line.strip())
                            record_cloud_usage(usage.get("input_tokens", 0), usage.get("output_tokens", 0), usage.get("model", decision.model))
                        except Exception:
                            pass
                        continue
                    try:
                        ev = json.loads(line.strip())
                        if ev.get("t") == "text":
                            full_response += ev.get("c", "")
                        elif ev.get("t") == "done":
                            _append_to_history(req.message, full_response)
                            yield json.dumps({"t": "done", "turns": len(_conversation_history) // 2}) + "\n"
                            continue
                    except (json.JSONDecodeError, ValueError):
                        pass
                    yield line
                return

            # Build messages for local model
            if rag_context:
                local_content = f"{rag_context}\n{req.message}"
            else:
                local_content = req.message
            local_messages = _get_history() + [{"role": "user", "content": local_content}]

            gathered = []  # list of {"tool": name, "args": args, "result": text}

            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    for round_num in range(MAX_TOOL_ROUNDS):
                        data, err, used_model, is_emergency = await _inference_with_emergency_fallback(
                            client, local_model, local_messages, TOOLS, backup,
                            num_gpu=decision.num_gpu, num_ctx=decision.num_ctx
                        )

                        if data is None:
                            logger.warning(f"Split Phase 1 failed: {err}")
                            break

                        msg = data.get("message", {})
                        tool_calls = msg.get("tool_calls")
                        content = msg.get("content", "")

                        # Text-as-tool fallback
                        if not tool_calls and content:
                            parsed = try_parse_text_tool_call(content)
                            if parsed:
                                name, args = parsed
                                tool_calls = [{"function": {"name": name, "arguments": args}}]
                                msg["tool_calls"] = tool_calls
                                msg["content"] = ""

                        if not tool_calls:
                            break  # No more tools to call

                        local_messages.append(msg)
                        for tc in tool_calls:
                            fn = tc.get("function", {})
                            name = fn.get("name", "unknown")
                            args = fn.get("arguments", {})
                            yield json.dumps({"t": "tool", "n": name, "a": args}) + "\n"
                            result = await asyncio.to_thread(execute_tool, name, args)
                            summary = result[:300].replace("\n", " ").replace("\r", "")
                            if len(result) > 300:
                                summary += "..."
                            yield json.dumps({"t": "result", "n": name, "s": summary}) + "\n"
                            # Truncate large results before packaging for cloud
                            truncated = result[:8000]
                            if len(result) > 8000:
                                truncated += f"\n... [truncated, {len(result)} chars total]"
                            gathered.append({"tool": name, "args": args, "result": truncated})
                            local_messages.append({"role": "tool", "content": result})

            except Exception as e:
                logger.error(f"Split Phase 1 exception: {e}")
                yield json.dumps({"t": "retry", "n": "split", "c": "Phase 1 encountered an error, falling back to cloud..."}) + "\n"

            # ── Fallback: no tools called → standard cloud with tools ──
            if not gathered:
                yield json.dumps({"t": "split_phase", "phase": 0, "c": "No data to gather — sending to cloud..."}) + "\n"
                full_response = ""
                async for line in anthropic_stream(
                    message=req.message, model=decision.model,
                    rag_context=rag_context, tools=TOOLS, execute_tool_fn=execute_tool,
                    history=_get_history(),                 ):
                    if line.strip().startswith('{"t":"_usage"'):
                        try:
                            usage = json.loads(line.strip())
                            record_cloud_usage(usage.get("input_tokens", 0), usage.get("output_tokens", 0), usage.get("model", decision.model))
                        except Exception:
                            pass
                        continue
                    try:
                        ev = json.loads(line.strip())
                        if ev.get("t") == "text":
                            full_response += ev.get("c", "")
                        elif ev.get("t") == "done":
                            _append_to_history(req.message, full_response)
                            yield json.dumps({"t": "done", "turns": len(_conversation_history) // 2}) + "\n"
                            continue
                    except (json.JSONDecodeError, ValueError):
                        pass
                    yield line
                return

            # ── Phase 2: Cloud reasoning over gathered data ──
            yield json.dumps({"t": "split_phase", "phase": 2, "c": "Data gathered, reasoning in cloud..."}) + "\n"

            # Format gathered results into context
            split_context_parts = ["[SPLIT WORKLOAD — Local tool results]"]
            for g in gathered:
                split_context_parts.append(
                    f"[Tool: {g['tool']} | Args: {json.dumps(g['args'])}]\n{g['result']}"
                )
            split_context_parts.append("[END SPLIT CONTEXT]")
            split_context = "\n\n".join(split_context_parts)

            # Combine with RAG context if present
            combined_context = f"{rag_context}\n\n{split_context}" if rag_context else split_context

            full_response = ""
            async for line in anthropic_stream(
                message=req.message, model=decision.model,
                rag_context=combined_context, tools=None,
                split_mode=True,                 history=_get_history(),
            ):
                if line.strip().startswith('{"t":"_usage"'):
                    try:
                        usage = json.loads(line.strip())
                        record_cloud_usage(usage.get("input_tokens", 0), usage.get("output_tokens", 0), usage.get("model", decision.model))
                    except Exception:
                        pass
                    continue
                try:
                    ev = json.loads(line.strip())
                    if ev.get("t") == "text":
                        full_response += ev.get("c", "")
                    elif ev.get("t") == "done":
                        _append_to_history(req.message, full_response)
                        yield json.dumps({"t": "done", "turns": len(_conversation_history) // 2}) + "\n"
                        continue
                except (json.JSONDecodeError, ValueError):
                    pass
                yield line

        return StreamingResponse(stream_split(), media_type="application/x-ndjson")

    # ── ANTHROPIC PATH ──────────────────────────────────────
    if decision.backend == "anthropic":
        async def stream_cloud():
            yield json.dumps({"t": "route", **decision.to_dict()}) + "\n"

            if rag_context:
                chunk_count = rag_context.count("[Source:")
                yield json.dumps({"t": "rag", "n": chunk_count}) + "\n"

            full_response = ""

            cloud_message = req.message
            if req.voice_input:
                cloud_message = f"[Voice input — may contain transcription errors. Interpret the intended meaning.]\n{req.message}"

            async for line in anthropic_stream(
                message=cloud_message,
                model=decision.model,
                rag_context=rag_context,
                tools=TOOLS,
                execute_tool_fn=execute_tool,
                history=_get_history(),
                            ):
                # Intercept _usage events for cost tracking
                if line.strip().startswith('{"t":"_usage"'):
                    try:
                        usage = json.loads(line.strip())
                        record_cloud_usage(
                            usage.get("input_tokens", 0),
                            usage.get("output_tokens", 0),
                            usage.get("model", decision.model),
                        )
                    except Exception:
                        pass
                    continue  # Don't send _usage to frontend
                # Accumulate text for history
                try:
                    ev = json.loads(line.strip())
                    if ev.get("t") == "text":
                        full_response += ev.get("c", "")
                    elif ev.get("t") == "done":
                        _append_to_history(req.message, full_response)
                        yield json.dumps({"t": "done", "turns": len(_conversation_history) // 2}) + "\n"
                        continue
                except (json.JSONDecodeError, ValueError):
                    pass
                yield line

        return StreamingResponse(stream_cloud(), media_type="application/x-ndjson")

    # ── OLLAMA PATH (with tool calling) ─────────────────────
    voice_prefix = ""
    if req.voice_input:
        voice_prefix = "[Voice input — may contain transcription errors. Interpret the intended meaning.]\n"

    if rag_context:
        user_content = f"{voice_prefix}{rag_context}\n{req.message}"
    else:
        user_content = f"{voice_prefix}{req.message}" if voice_prefix else req.message

    messages = _get_smart_history(max_tokens=decision.num_ctx) + [{"role": "user", "content": user_content}]

    # Filter tools based on query domain and complexity
    relevant_tools = filter_tools(TOOLS, domain=decision.adapter_domain,
                                  tier=decision.tier, category=decision.query_category,
                                  query=req.message)

    # Track metadata for quality scoring and routing feedback
    _chat_metadata = {
        "tier": decision.tier, "domain": decision.adapter_domain or "general",
        "model": decision.model, "tool_calls": [], "tool_results": [],
        "react_used": False,
    }
    _chat_start_time = _time.time()

    async def stream_local():
        nonlocal messages

        yield json.dumps({"t": "route", **decision.to_dict()}) + "\n"

        if rag_context:
            chunk_count = rag_context.count("[Source:")
            yield json.dumps({"t": "rag", "n": chunk_count}) + "\n"

        # ── ReAct reasoning path for complex local queries ──
        if _is_react_eligible(decision):
            _chat_metadata["react_used"] = True
            yield json.dumps({"t": "react", "c": "Entering structured reasoning mode..."}) + "\n"
            async with httpx.AsyncClient(timeout=120) as react_client:
                final_text, tool_events = await _react_reasoning_loop(
                    react_client, decision.model, req.message,
                    rag_context, _get_smart_history(max_tokens=decision.num_ctx),
                    relevant_tools, execute_tool,
                    num_gpu=decision.num_gpu, num_ctx=decision.num_ctx
                )
                for ev in tool_events:
                    yield json.dumps(ev) + "\n"
                    if ev.get("t") == "tool":
                        _chat_metadata["tool_calls"].append(ev.get("n", ""))
                    if ev.get("t") == "result":
                        _chat_metadata["tool_results"].append({"name": ev.get("n", ""), "success": True})
                yield json.dumps({"t": "text", "c": final_text}) + "\n"
                _chat_metadata["latency_ms"] = (_time.time() - _chat_start_time) * 1000
                _append_to_history(req.message, final_text, chat_metadata=_chat_metadata)
                yield json.dumps({"t": "done", "turns": len(_conversation_history) // 2}) + "\n"
            return

        # Track which model is active (may change if emergency backup engages)
        active_model = decision.model
        backup = decision.backup_model
        emergency_active = False

        rounds = 0
        async with httpx.AsyncClient(timeout=120) as client:
            while rounds < MAX_TOOL_ROUNDS:
                rounds += 1

                # ── Inference with retry-first, backup-last ──
                data, err, used_model, is_emergency = await _inference_with_emergency_fallback(
                    client, active_model, messages, TOOLS, backup if not emergency_active else None,
                    num_gpu=decision.num_gpu, num_ctx=decision.num_ctx
                )

                if is_emergency and not emergency_active:
                    emergency_active = True
                    active_model = used_model
                    backup = None  # don't cascade further
                    yield json.dumps({
                        "t": "emergency",
                        "c": f"PRIMARY MODEL OFFLINE — Running on backup: {used_model}"
                    }) + "\n"

                if data is None:
                    # User-friendly error messages
                    err_l = (err or "").lower()
                    if "timed out" in err_l:
                        friendly = "Model took too long to respond. Try a shorter question or check if Ollama is running."
                    elif "connect" in err_l or "refused" in err_l:
                        friendly = "Cannot reach Ollama. Make sure it's running (ollama serve)."
                    elif "all models failed" in err_l:
                        friendly = "All models are offline. Check Ollama and model availability."
                    else:
                        friendly = "Backend error. Please try again."
                    yield json.dumps({"t": "error", "c": friendly}) + "\n"
                    yield json.dumps({"t": "done"}) + "\n"
                    return

                msg = data.get("message", {})
                tool_calls = msg.get("tool_calls")
                content = msg.get("content", "")

                if not tool_calls and content:
                    parsed = try_parse_text_tool_call(content)
                    if parsed:
                        name, args = parsed
                        tool_calls = [{"function": {"name": name, "arguments": args}}]
                        msg["tool_calls"] = tool_calls
                        msg["content"] = ""

                if not tool_calls:
                    if content:
                        chunk_size = 4
                        for i in range(0, len(content), chunk_size):
                            yield json.dumps({"t": "text", "c": content[i:i+chunk_size]}) + "\n"
                    _append_to_history(req.message, content)
                    yield json.dumps({"t": "done", "turns": len(_conversation_history) // 2}) + "\n"
                    return

                messages.append(msg)
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "unknown")
                    args = fn.get("arguments", {})
                    yield json.dumps({"t": "tool", "n": name, "a": args}) + "\n"
                    try:
                        result = await asyncio.to_thread(execute_tool, name, args)
                    except Exception as tool_err:
                        safe_errors.log_exception(logger, "Tool execution crashed", tool_err)
                        result = "ERROR: Tool execution crashed"
                    # If tool errored, retry once with sanitized args
                    if result.startswith("ERROR:") and name in ("run_shell", "read_file", "write_file"):
                        yield json.dumps({"t": "retry", "n": name, "c": "Retrying with adjusted parameters..."}) + "\n"
                        sanitized_args = dict(args)
                        if "path" in sanitized_args:
                            sanitized_args["path"] = os.path.normpath(sanitized_args["path"])
                        if "timeout" in sanitized_args and name == "run_shell":
                            sanitized_args["timeout"] = min(int(sanitized_args.get("timeout", 15)), 30)
                        try:
                            result2 = await asyncio.to_thread(execute_tool, name, sanitized_args)
                            if not result2.startswith("ERROR:"):
                                result = result2
                        except Exception:
                            pass  # keep original error
                    summary = result[:300].replace("\n", " ").replace("\r", "")
                    if len(result) > 300:
                        summary += "..."
                    yield json.dumps({"t": "result", "n": name, "s": summary}) + "\n"
                    messages.append({"role": "tool", "content": result})

            # ── Max tool rounds — final response ──
            data, err, used_model, is_emergency = await _inference_with_emergency_fallback(
                client, active_model, messages, None, backup if not emergency_active else None,
                num_gpu=decision.num_gpu, num_ctx=decision.num_ctx
            )
            if is_emergency and not emergency_active:
                yield json.dumps({
                    "t": "emergency",
                    "c": f"PRIMARY MODEL OFFLINE — Running on backup: {used_model}"
                }) + "\n"
            if data is not None:
                content = data.get("message", {}).get("content", "Max tool rounds reached.")
                for i in range(0, len(content), 4):
                    yield json.dumps({"t": "text", "c": content[i:i+4]}) + "\n"
                _append_to_history(req.message, content)
            else:
                err_l = (err or "").lower()
                if "timed out" in err_l:
                    friendly = "Model took too long to respond. Try a shorter question or check if Ollama is running."
                elif "connect" in err_l or "refused" in err_l:
                    friendly = "Cannot reach Ollama. Make sure it's running (ollama serve)."
                elif "all models failed" in err_l:
                    friendly = "All models are offline. Check Ollama and model availability."
                else:
                    friendly = "Backend error. Please try again."
                yield json.dumps({"t": "error", "c": friendly}) + "\n"
            yield json.dumps({"t": "done", "turns": len(_conversation_history) // 2}) + "\n"

    return StreamingResponse(stream_local(), media_type="application/x-ndjson")


# ── CONVERSATION HISTORY ENDPOINTS ─────────────────────────────────────

@app.delete("/chat/history")
def clear_chat_history():
    """Clear conversation history."""
    _conversation_history.clear()
    try:
        db_clear_history()
    except Exception as e:
        logger.warning(f"Failed to clear DB history: {e}")
    return {"ok": True, "message": "History cleared"}

@app.get("/chat/history")
def get_chat_history():
    """Get conversation history metadata."""
    turns = len(_conversation_history) // 2
    return {"turns": turns, "max_turns": HISTORY_MAX_TURNS}


# ── ROUTER CONFIG ENDPOINTS ────────────────────────────────────────────

@app.get("/router/status")
def router_status():
    """Get current routing configuration and status."""
    return {
        "cloud_enabled": os.getenv("CLOUD_ENABLED", "true").lower() == "true",
        "cloud_available": is_cloud_available_sync(),
        "gpu_utilization": get_gpu_utilization(),
        "gpu_threshold": int(os.getenv("GPU_THRESHOLD", "70")),
        "has_api_key": bool(os.getenv("ANTHROPIC_API_KEY", "").strip()),
        "sonnet_model": os.getenv("ANTHROPIC_SONNET_MODEL", "claude-sonnet-4-20250514"),
        "opus_model": os.getenv("ANTHROPIC_OPUS_MODEL", "claude-opus-4-20250514"),
        "local_model": os.getenv("DELTAI_MODEL", "deltai"),
        "strong_model": os.getenv("DELTAI_STRONG_MODEL", ""),
    }

class CloudToggle(BaseModel):
    enabled: bool

@app.post("/router/cloud")
def toggle_cloud(req: CloudToggle):
    """Toggle cloud on/off at runtime."""
    os.environ["CLOUD_ENABLED"] = str(req.enabled).lower()
    return {"cloud_enabled": req.enabled}


# ── CLOUD BUDGET ENDPOINTS ────────────────────────────────────────────

@app.get("/budget/status")
def budget_endpoint():
    """Cloud cost budget status — daily spend, limit, remaining."""
    return get_budget_status()


# ── MEMORY ENDPOINTS ────────────────────────────────────────────────────

@app.get("/memory/stats")
def memory_stats_endpoint():
    if not RAG_AVAILABLE:
        return {"error": "RAG not available"}
    try:
        stats = get_memory_stats()
        stats["watcher_active"] = watcher_running()
        return stats
    except Exception as e:
        return {"error": safe_errors.public_error_detail(e)}

@app.post("/memory/ingest")
def memory_ingest_endpoint():
    if not RAG_AVAILABLE:
        return {"error": "RAG not available"}
    try:
        results = ingest_all()
        return {
            "results": results,
            "ingested": len([r for r in results if r["status"] == "ok"]),
            "skipped": len([r for r in results if r["status"] == "skipped"]),
            "errors": len([r for r in results if r["status"] == "error"]),
        }
    except Exception as e:
        return {"error": safe_errors.public_error_detail(e)}

@app.get("/memory/files")
def memory_files_endpoint():
    """Get detailed per-file info from ChromaDB."""
    if not RAG_AVAILABLE:
        return {"error": "RAG not available", "files": []}
    try:
        files = get_file_details()
        return {"files": files}
    except Exception as e:
        return {"error": safe_errors.public_error_detail(e), "files": []}

@app.delete("/memory/files/{filepath:path}")
def memory_delete_file(filepath: str):
    """Remove a specific file's chunks from ChromaDB."""
    if not RAG_AVAILABLE:
        return {"error": "RAG not available"}
    try:
        full_path = os.path.realpath(os.path.join(KNOWLEDGE_PATH, filepath))
        if not full_path.startswith(os.path.realpath(KNOWLEDGE_PATH)):
            return {"status": "error", "reason": "Invalid path — outside knowledge directory"}
        result = remove_file(full_path)
        return result
    except Exception as e:
        return {"status": "error", "reason": safe_errors.public_error_detail(e)}


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
                except asyncio.TimeoutError:
                    break

            if not batch or not RAG_AVAILABLE:
                continue

            # Process batch
            start = _time.time()
            items_for_batch = [
                {"source": it["source"], "context": it["context"],
                 "ttl": it["ttl"], "tags": it["tags"]}
                for it in batch
            ]
            try:
                result = await asyncio.to_thread(ingest_context_batch, items_for_batch)
                _ingest_metrics["processed"] += len(batch)
                elapsed = (_time.time() - start) * 1000
                _ingest_metrics["avg_latency_ms"] = (
                    _ingest_metrics["avg_latency_ms"] * 0.8 + elapsed * 0.2
                )
            except Exception as e:
                _ingest_metrics["errors"] += len(batch)
                logger.warning(f"Ingest pipeline batch failed: {e}")

            _ingest_metrics["last_flush"] = _time.time()

            # Forward alerts
            for it in batch:
                if "alert" in it.get("tags", []):
                    alert = {
                        "source": it["source"],
                        "context": it["context"],
                        "tags": it["tags"],
                        "ts": _time.time(),
                        "priority": "critical" if "critical" in it["tags"] else "high" if "high" in it["tags"] else "normal",
                    }
                    _recent_alerts.append(alert)
                    await _broadcast_alert(alert)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Ingest pipeline error: {e}")
            await asyncio.sleep(1)


# ── INGEST CONNECTOR ──────────────────────────────────────────────────
# External services push structured context into deltai's RAG memory.

class IngestRequest(BaseModel):
    source: str               # identifies the sender (e.g., "lmu-telemetry")
    context: str              # human-readable content to embed
    ttl: int = 0              # seconds until auto-expiry (0 = permanent)
    tags: list[str] = []      # optional filtering tags


@app.post("/ingest")
async def ingest_endpoint(req: IngestRequest):
    """
    Ingest structured context from an external service into ChromaDB.
    This is deltai's connector — any service can push context here.
    When the async pipeline is active, returns immediately after queuing.
    """
    if not RAG_AVAILABLE:
        return {"error": "RAG not available"}
    if not req.source or not req.source.strip():
        return {"status": "error", "reason": "source is required"}
    if not req.context or not req.context.strip():
        return {"status": "error", "reason": "context is required"}

    # Try async pipeline first (non-blocking)
    if _ingest_queue is not None:
        try:
            _ingest_queue.put_nowait({
                "source": req.source.strip(),
                "context": req.context.strip(),
                "ttl": req.ttl,
                "tags": req.tags,
            })
            _ingest_metrics["queued"] += 1
            # Immediate alert forwarding (don't wait for pipeline)
            if "alert" in req.tags:
                alert = {
                    "source": req.source,
                    "context": req.context,
                    "tags": req.tags,
                    "ts": _time.time(),
                    "priority": "critical" if "critical" in req.tags else "high" if "high" in req.tags else "normal",
                }
                _recent_alerts.append(alert)
                await _broadcast_alert(alert)
            return {"status": "queued", "queue_depth": _ingest_queue.qsize()}
        except asyncio.QueueFull:
            return JSONResponse(
                status_code=429,
                content={"status": "error", "reason": "Ingest queue full — backpressure"},
            )

    # Fallback: synchronous ingest
    try:
        result = await asyncio.to_thread(
            ingest_context,
            source=req.source.strip(),
            context=req.context.strip(),
            ttl=req.ttl,
            tags=req.tags,
        )
        if "alert" in req.tags:
            alert = {
                "source": req.source,
                "context": req.context,
                "tags": req.tags,
                "ts": _time.time(),
                "priority": "critical" if "critical" in req.tags else "high" if "high" in req.tags else "normal",
            }
            _recent_alerts.append(alert)
            await _broadcast_alert(alert)
        return result
    except Exception as e:
        return {"status": "error", "reason": safe_errors.public_error_detail(e)}


class IngestBatchRequest(BaseModel):
    items: list[dict]  # each: {"source": str, "context": str, "ttl": int, "tags": list[str]}


@app.post("/ingest/batch")
async def ingest_batch_endpoint(req: IngestBatchRequest):
    """Batch ingest multiple context items. Max 100 per batch."""
    if not RAG_AVAILABLE:
        return {"error": "RAG not available"}
    try:
        result = await asyncio.to_thread(ingest_context_batch, req.items)
        # Forward any alerts
        for item in req.items:
            tags = item.get("tags", [])
            if "alert" in tags:
                alert = {
                    "source": item.get("source", ""),
                    "context": item.get("context", ""),
                    "tags": tags,
                    "ts": _time.time(),
                    "priority": "critical" if "critical" in tags else "high" if "high" in tags else "normal",
                }
                _recent_alerts.append(alert)
                await _broadcast_alert(alert)
        return result
    except Exception as e:
        return {"status": "error", "reason": safe_errors.public_error_detail(e)}


@app.get("/ingest/pipeline/status")
def ingest_pipeline_status():
    """Get ingest pipeline queue metrics."""
    return {
        "pipeline_active": _ingest_queue is not None,
        "queue_depth": _ingest_queue.qsize() if _ingest_queue else 0,
        "queue_max": _INGEST_QUEUE_MAX,
        **_ingest_metrics,
    }


@app.post("/ingest/cleanup")
def ingest_cleanup():
    """Manually trigger TTL cleanup of expired ingested entries."""
    if not RAG_AVAILABLE:
        return {"error": "RAG not available"}
    try:
        return cleanup_expired()
    except Exception as e:
        return {"status": "error", "reason": safe_errors.public_error_detail(e)}


# ── HIERARCHICAL MEMORY ENDPOINTS ─────────────────────────────────────

@app.post("/memory/compact")
async def memory_compact():
    """Manually trigger warm→cold memory compaction."""
    if not RAG_AVAILABLE:
        return {"error": "RAG not available"}
    try:
        result = await asyncio.to_thread(compact_warm_to_cold)
        return result
    except Exception as e:
        return {"status": "error", "reason": safe_errors.public_error_detail(e)}


@app.get("/memory/cold/stats")
def cold_memory_stats():
    """Get cold tier storage statistics."""
    if not RAG_AVAILABLE:
        return {"error": "RAG not available"}
    try:
        return get_cold_stats()
    except Exception as e:
        return {"error": safe_errors.public_error_detail(e)}


# ── KNOWLEDGE GAP ENDPOINTS ───────────────────────────────────────────

@app.get("/knowledge/gaps")
def knowledge_gaps():
    """Get unresolved knowledge gaps."""
    try:
        gaps = get_unresolved_gaps(limit=50)
        return {"gaps": gaps, "count": len(gaps)}
    except Exception as e:
        return {"error": safe_errors.public_error_detail(e)}


@app.post("/knowledge/gaps/{gap_id}/resolve")
def resolve_gap(gap_id: int):
    """Mark a knowledge gap as resolved."""
    try:
        resolve_knowledge_gap(gap_id)
        return {"status": "ok", "resolved": gap_id}
    except Exception as e:
        return {"status": "error", "reason": safe_errors.public_error_detail(e)}


# ── TRAINING INTELLIGENCE ENDPOINTS ──────────────────────────────────

@app.get("/training/weaknesses")
def training_weaknesses():
    """Identify domains where deltai performs poorly."""
    if not TRAINING_AVAILABLE:
        return {"error": "Training not available"}
    try:
        from training import identify_weak_domains
        return {"weaknesses": identify_weak_domains()}
    except Exception as e:
        return {"error": safe_errors.public_error_detail(e)}


@app.post("/training/improve/{domain}")
async def training_improve(domain: str):
    """Trigger an improvement cycle for a weak domain."""
    if not TRAINING_AVAILABLE:
        return {"error": "Training not available"}
    try:
        from training import run_improvement_cycle
        result = await asyncio.to_thread(run_improvement_cycle, domain)
        return result
    except Exception as e:
        return {"status": "error", "reason": safe_errors.public_error_detail(e)}


# ── WEBSOCKET ALERTS ─────────────────────────────────────────────────

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket endpoint for proactive alerts (e.g. tagged ingest notifications)."""
    await websocket.accept()
    _alert_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep alive
    except WebSocketDisconnect:
        pass
    finally:
        try:
            _alert_clients.remove(websocket)
        except ValueError:
            pass


@app.get("/alerts/recent")
def recent_alerts():
    """Get the last 20 alerts."""
    return {"alerts": list(_recent_alerts)}


# ── HEALTH EVENT BUS ENDPOINTS ────────────────────────────────────────

@app.websocket("/ws/health")
async def websocket_health(websocket: WebSocket):
    """WebSocket endpoint for real-time health state events."""
    await websocket.accept()
    _health_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        try:
            _health_clients.remove(websocket)
        except ValueError:
            pass


@app.get("/health/events")
def health_events(limit: int = 50):
    """Get recent health events."""
    return {"events": list(_health_events)[-limit:]}


# ── SELF-HEAL STATUS ─────────────────────────────────────────────────

@app.get("/self-heal/status")
def self_heal_status():
    """AI self-heal loop status."""
    heal_actions = [a for a in _resource_state["actions_taken"] if "Self-heal" in a.get("action", "")]
    return {
        "enabled": _SELF_HEAL_ENABLED,
        "interval_sec": _SELF_HEAL_INTERVAL,
        "model": _SELF_HEAL_MODEL,
        "idle": _is_idle(),
        "circuit_breaker": _circuit_breaker["state"],
        "recent_actions": heal_actions[-10:],
    }


# ── RESOURCE MANAGEMENT STATUS ────────────────────────────────────────

@app.get("/resources/status")
def resource_status():
    """Resource self-manager status — VRAM, circuit breaker, auto-recovery."""
    vram_free = get_vram_free_mb()
    decline_rate = _predict_vram_decline()
    gpu_temp = _get_gpu_temp()
    return {
        "resource_manager": {
            "vram_free_mb": vram_free,
            "vram_warnings": _resource_state["vram_warnings"],
            "vram_decline_rate_mb_s": round(decline_rate, 1),
            "vram_prediction": "declining" if decline_rate > 50 else "stable" if abs(decline_rate) < 20 else "recovering",
            "gpu_temp_c": gpu_temp,
            "process_priority": "below_normal" if _resource_state["priority_lowered"] else "normal",
            "ollama_failures": _resource_state["ollama_failures"],
            "watcher_restarts": _resource_state["watcher_restarts"],
            "last_vram_action": _resource_state["last_vram_action"] or None,
            "last_recovery": _resource_state["last_recovery"] or None,
            "recent_actions": _resource_state["actions_taken"][-10:],
        },
        "circuit_breaker": {
            "state": _circuit_breaker["state"],
            "failures": _circuit_breaker["failures"],
            "backoff_sec": _circuit_breaker["backoff_sec"],
        },
    }


# ── SUBSYSTEM HEALTH ──────────────────────────────────────────────────

@app.get("/api/health")
async def system_health():
    """Returns status of all deltai subsystems for the header health monitor."""
    results = {}

    # Ollama
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags", timeout=3)
            results["ollama"] = "online" if r.status_code == 200 else "down"
    except Exception:
        results["ollama"] = "down"

    # FastAPI — always online if this endpoint responds
    results["fastapi"] = "online"

    # ChromaDB / RAG
    if RAG_AVAILABLE:
        try:
            stats = get_memory_stats()
            results["chromadb"] = "online" if stats else "degraded"
        except Exception:
            results["chromadb"] = "degraded"
    else:
        results["chromadb"] = "down"

    # Smart Router
    try:
        model, cpu_only, backup = _pick_local_model()
        results["router"] = "local" if model else "down"
    except Exception:
        results["router"] = "down"

    # File Watcher
    if RAG_AVAILABLE:
        try:
            results["watcher"] = "active" if watcher_running() else "stopped"
        except Exception:
            results["watcher"] = "stopped"
    else:
        results["watcher"] = "stopped"

    # Tool System
    results["tools"] = "online" if len(TOOLS) > 0 else "down"

    # Anthropic API
    results["anthropic"] = "ready" if is_cloud_available_sync() else "standby"

    # Voice
    if VOICE_AVAILABLE:
        try:
            vs = get_voice_status()
            results["voice"] = "online" if vs.get("enabled") else "disabled"
        except Exception:
            results["voice"] = "degraded"
    else:
        results["voice"] = "unavailable"

    # Telemetry API (conditional — only when configured)
    if TELEMETRY_API_URL:
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                r = await client.get(f"{TELEMETRY_API_URL}/health")
                results["telemetry"] = "online" if r.status_code == 200 else "down"
        except Exception:
            results["telemetry"] = "down"

    return results


# ── EMERGENCY BACKUP STATUS ────────────────────────────────────────────

@app.get("/backup/status")
def backup_status():
    """Emergency backup system status."""
    backup_enabled = os.getenv("BACKUP_ENABLED", "true").lower() in ("true", "1", "yes")
    strong = os.getenv("DELTAI_STRONG_MODEL", "deltai-qwen14b")
    default = os.getenv("DELTAI_MODEL", "deltai-qwen3b")
    return {
        "enabled": backup_enabled,
        "mapping": {
            strong: os.getenv("DELTAI_BACKUP_STRONG_MODEL", "deltai-nemo"),
            default: os.getenv("DELTAI_BACKUP_MODEL", "deltai-fallback"),
        },
        "models": _backup_health_status,
        "last_activated": _backup_last_activated if _backup_last_activated > 0 else None,
        "health_interval_sec": int(os.getenv("BACKUP_HEALTH_INTERVAL_SEC", "3600")),
        "max_retries_before_backup": BACKUP_MAX_RETRIES,
    }


# ── TRAINING ENDPOINTS ─────────────────────────────────────────────────

class DatasetCreate(BaseModel):
    name: str

class ExampleAdd(BaseModel):
    input: Optional[str] = ""
    output: str
    instruction: Optional[str] = ""
    category: str = "general"

class ExportRequest(BaseModel):
    format: str = "alpaca"

@app.get("/training/datasets")
def training_list_datasets():
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable", "datasets": []}
    return {"datasets": list_datasets()}

@app.post("/training/datasets")
def training_create_dataset(req: DatasetCreate):
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    return create_dataset(req.name)

@app.delete("/training/datasets/{name}")
def training_delete_dataset(name: str):
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    return delete_dataset(name)

@app.get("/training/datasets/{name}")
def training_get_dataset(name: str):
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable", "examples": []}
    return get_dataset(name)

@app.post("/training/datasets/{name}/add")
def training_add_example(name: str, req: ExampleAdd):
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    # Combine instruction + input for Alpaca-style training
    input_text = req.input or ""
    if req.instruction:
        input_text = req.instruction + ("\n" + input_text if input_text else "")
    return add_example(name, input_text, req.output, req.category)

@app.delete("/training/datasets/{name}/{idx}")
def training_remove_example(name: str, idx: int):
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    return remove_example(name, idx)

@app.post("/training/datasets/{name}/export")
def training_export_dataset(name: str, req: ExportRequest):
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    return export_dataset(name, req.format)

@app.get("/training/status")
def training_status():
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    return get_training_status()


class TrainingStart(BaseModel):
    dataset: str
    base_model: str = "deltai-qwen3b"
    output_model: str | None = None
    mode: str = "auto"  # "lora", "fewshot", "auto", or "distill"
    teacher_dataset: str | None = None      # for distill mode
    replay_datasets: list[str] | None = None  # for distill mode


@app.post("/training/start")
def training_start(req: TrainingStart):
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    return start_training(
        dataset_name=req.dataset,
        base_model=req.base_model,
        output_model=req.output_model,
        mode=req.mode,
        teacher_dataset=req.teacher_dataset,
        replay_datasets=req.replay_datasets,
    )


@app.post("/training/stop")
def training_stop():
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    return stop_training()


@app.get("/training/lora/status")
def training_lora_status():
    """Check if LoRA training dependencies are installed."""
    if not TRAINING_AVAILABLE:
        return {"available": False, "reason": "Training system unavailable"}
    from training import check_lora_deps
    ok, reason = check_lora_deps()
    return {"available": ok, "reason": reason if not ok else "ready"}


class ABEvalRequest(BaseModel):
    model_a: str
    model_b: str
    dataset: str
    max_examples: int = 20


@app.post("/training/eval/ab")
def training_ab_eval(req: ABEvalRequest):
    """Run A/B evaluation comparing two models on a dataset."""
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    from training import run_ab_eval
    return run_ab_eval(req.model_a, req.model_b, req.dataset, req.max_examples)


# ── ADAPTER SURGERY ENDPOINTS ────────────────────────────────────────
# Modular LoRA augmentation slots: train, merge, evaluate, promote.

@app.get("/adapters")
def adapters_list(domain: str = None, status: str = None):
    """List all registered adapters, optionally filtered."""
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    from training import list_adapters
    return {"adapters": list_adapters(domain=domain, status=status)}


@app.get("/adapters/active")
def adapters_active():
    """Get the current active adapter map."""
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    from training import get_active_adapters
    return {"active": get_active_adapters()}


@app.get("/adapters/{name}")
def adapters_get(name: str):
    """Get details for a specific adapter."""
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    from training import get_adapter
    result = get_adapter(name)
    if not result:
        return JSONResponse({"error": f"Adapter not found: {name}"}, status_code=404)
    return result


class AdapterTrainRequest(BaseModel):
    domain: str
    dataset: str = None
    freeze_layers: int = None
    lr: float = None
    epochs: int = None


@app.post("/adapters/train")
def adapters_train(req: AdapterTrainRequest):
    """Start domain-targeted adapter training for an augmentation slot."""
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    from training import start_domain_training
    return start_domain_training(
        domain=req.domain,
        dataset_name=req.dataset,
        freeze_layers=req.freeze_layers,
        lr_override=req.lr,
        epochs_override=req.epochs,
    )


class AdapterMergeRequest(BaseModel):
    adapters: list[str] = None
    method: str = "ties"
    density: float = 0.5
    output_model: str = None


@app.post("/adapters/merge")
def adapters_merge(req: AdapterMergeRequest):
    """Merge active adapters into a single production GGUF model via TIES."""
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    from training import merge_adapters
    return merge_adapters(
        adapter_names=req.adapters,
        method=req.method,
        density=req.density,
        output_model=req.output_model,
    )


@app.post("/adapters/eval/{name}")
def adapters_eval(name: str, dataset: str = None, max_examples: int = 20):
    """Evaluate an adapter against the baseline model."""
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    from training import eval_adapter
    return eval_adapter(name, eval_dataset=dataset, max_examples=max_examples)


@app.post("/adapters/promote/{name}")
def adapters_promote(name: str):
    """Promote an adapter to the active slot for its domain."""
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    from training import get_adapter, set_active_adapter, update_adapter
    info = get_adapter(name)
    if not info:
        return JSONResponse({"error": f"Adapter not found: {name}"}, status_code=404)
    domain = info.get("domain")
    result = set_active_adapter(domain, name)
    if result.get("status") == "ok":
        update_adapter(name, promoted=True)
    return result


class AdapterRollbackRequest(BaseModel):
    domain: str
    version: int = None


@app.post("/adapters/rollback")
def adapters_rollback(req: AdapterRollbackRequest):
    """Roll back a domain to a previous adapter version."""
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    from training import rollback_adapter
    return rollback_adapter(req.domain, target_version=req.version)


@app.delete("/adapters/{name}")
def adapters_delete(name: str, delete_files: bool = False):
    """Remove an adapter from the registry."""
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    from training import remove_adapter
    return remove_adapter(name, delete_files=delete_files)


# ── KNOWLEDGE DISTILLATION ENDPOINTS ─────────────────────────────────

class TeacherGenerateRequest(BaseModel):
    queries: list[str]
    teacher: str = "local14b"  # "local14b" or "anthropic"
    dataset: str = "deltai-teacher"
    category: str = "distilled"


@app.post("/training/generate-teacher")
def training_generate_teacher(req: TeacherGenerateRequest):
    """Generate training data using a teacher model (14B local or Anthropic API)."""
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    if req.teacher not in ("local14b", "anthropic"):
        return {"error": f"Unknown teacher: {req.teacher}. Use 'local14b' or 'anthropic'."}
    if not req.queries:
        return {"error": "No queries provided"}
    from training import generate_teacher_data
    return generate_teacher_data(
        queries=req.queries,
        teacher=req.teacher,
        dataset_name=req.dataset,
        category=req.category,
    )


class BlendRequest(BaseModel):
    sources: list[dict]  # [{"dataset":"name", "weight":0.3, "max_examples":100}]
    output: str


@app.post("/training/blend")
def training_blend(req: BlendRequest):
    """Blend multiple datasets with weighted sampling for distillation training."""
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    if not req.sources:
        return {"error": "No source datasets provided"}
    if not req.output:
        return {"error": "No output dataset name provided"}
    from training import blend_datasets
    return blend_datasets(sources=req.sources, output_name=req.output)


class RetentionRequest(BaseModel):
    model: str
    baseline: str = "deltai-qwen3b"
    min_pass_rate: float = 0.7


@app.post("/training/verify-retention")
def training_verify_retention(req: RetentionRequest):
    """Verify a trained model hasn't lost base capabilities."""
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    from training import verify_retention
    return verify_retention(
        model_name=req.model,
        baseline_model=req.baseline,
        min_pass_rate=req.min_pass_rate,
    )


# ── VOICE ENDPOINTS ──────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    rate: Optional[str] = None
    pitch: Optional[str] = None


@app.get("/voice/status")
def voice_status():
    """Voice subsystem status."""
    if not VOICE_AVAILABLE:
        return {"enabled": False, "error": "Voice module not loaded",
                "stt": {"status": "unavailable"}, "tts": {"status": "unavailable"}}
    return get_voice_status()


@app.post("/voice/warm")
async def voice_warm():
    """Pre-warm the Whisper STT model. Call when voice mode activates to avoid cold start."""
    if not VOICE_AVAILABLE:
        return {"ok": False, "error": "Voice module not loaded"}
    try:
        await asyncio.to_thread(transcribe_audio, b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
        return {"ok": True, "stt": "warmed"}
    except Exception as e:
        return {"ok": False, "error": safe_errors.public_error_detail(e)}


@app.post("/voice/stt")
async def voice_stt(request: Request):
    """Transcribe uploaded audio to text.

    Accepts raw audio bytes (WAV, MP3, FLAC) via POST body.
    """
    if not VOICE_AVAILABLE:
        return {"error": "Voice module not available"}
    body = await request.body()
    if not body:
        return {"error": "No audio data received"}
    if len(body) > 25 * 1024 * 1024:  # 25MB limit
        return {"error": "Audio file too large (max 25MB)"}
    result = await asyncio.to_thread(transcribe_audio, body)
    return result


@app.post("/voice/tts")
async def voice_tts(req: TTSRequest):
    """Convert text to speech audio. Returns audio bytes."""
    if not VOICE_AVAILABLE:
        return JSONResponse({"error": "Voice module not available"}, status_code=503)
    result = await synthesize_speech(req.text, voice=req.voice, rate=req.rate, pitch=req.pitch)
    if "error" in result:
        logger.error("Voice TTS failed (detail omitted from logs and response)")
        return JSONResponse({"error": "Voice synthesis failed"}, status_code=500)
    return Response(
        content=result["audio"],
        media_type=f"audio/{result['format']}",
        headers={
            "Content-Disposition": f"inline; filename=deltai_speech.{result['format']}",
            "X-Duration-Estimate": str(result.get("duration_estimate", 0)),
        },
    )


@app.post("/voice/chat")
async def voice_chat(request: Request, voice: str = None, rate: str = None):
    """Full voice loop: transcribe audio → chat → synthesize response.

    POST raw audio → returns JSON with transcription + base64 audio response.
    Optional query params: voice (TTS voice ID), rate (TTS speed e.g. "+15%").
    """
    if not VOICE_AVAILABLE:
        return {"error": "Voice module not available"}
    body = await request.body()
    if not body:
        return {"error": "No audio data"}

    # Step 1: Transcribe
    stt_result = await asyncio.to_thread(transcribe_audio, body)
    if "error" in stt_result or not stt_result.get("text"):
        return {"error": stt_result.get("error", "No speech detected"), "stt": stt_result}

    user_text = stt_result["text"]

    # Step 2: Chat
    decision = await route(message=user_text, force_cloud=False, force_local=False)

    greeting_response = _check_greeting(user_text)
    if greeting_response:
        response_text = greeting_response
    else:
        rag_context = build_rag_context(user_text)
        voice_hint = "[Voice input — may contain transcription errors. Interpret the intended meaning.]\n"
        user_content = f"{voice_hint}{rag_context}\n{user_text}" if rag_context else f"{voice_hint}{user_text}"
        msg_list = _get_history() + [{"role": "user", "content": user_content}]

        async with httpx.AsyncClient(timeout=120) as client:
            data, err, used_model, is_emergency = await _inference_with_emergency_fallback(
                client, decision.model, msg_list, TOOLS, decision.backup_model,
                num_gpu=decision.num_gpu, num_ctx=decision.num_ctx
            )
        if data is None:
            response_text = "Systems encountered an error. Try again."
        else:
            response_text = data.get("message", {}).get("content", "No response generated.")

        _append_to_history(user_text, response_text)

    # Step 3: TTS (use custom voice/rate if provided via query params)
    tts_result = await synthesize_speech(response_text, voice=voice, rate=rate)
    audio_b64 = None
    if "audio" in tts_result:
        audio_b64 = base64.b64encode(tts_result["audio"]).decode("ascii")

    return {
        "stt": {"text": user_text, "language": stt_result.get("language", "en")},
        "response": response_text,
        "tts": {
            "audio_base64": audio_b64,
            "format": tts_result.get("format", "mp3"),
            "duration_estimate": tts_result.get("duration_estimate", 0),
        } if audio_b64 else {"error": tts_result.get("error", "TTS failed")},
        "route": decision.to_dict(),
    }


# ── NEW VOICE PIPELINE (Phase 1 — Piper + RVC + Effects) ─────────────────

try:
    from voice import speak as voice_speak, configure as voice_configure
    from voice.voice_config import DEFAULT_CONFIG as _voice_cfg, VoiceConfig
    VOICE_PIPELINE_AVAILABLE = True
except ImportError:
    VOICE_PIPELINE_AVAILABLE = False


@app.post("/api/voice/speak")
async def api_voice_speak(req: dict):
    """Speak text through the new voice pipeline (Piper → RVC → effects → playback)."""
    if not VOICE_PIPELINE_AVAILABLE:
        return JSONResponse({"error": "Voice pipeline module not available"}, status_code=503)
    text = req.get("text", "")
    priority = req.get("priority", "normal")
    if not text.strip():
        return {"error": "Empty text"}
    try:
        await voice_speak(text, priority=priority)
        return {"ok": True, "text": text, "priority": priority}
    except Exception as e:
        return JSONResponse({"error": safe_errors.public_error_detail(e)}, status_code=500)


@app.get("/api/voice/config")
async def api_voice_config_get():
    """Return current voice pipeline configuration."""
    if not VOICE_PIPELINE_AVAILABLE:
        return JSONResponse({"error": "Voice pipeline not available"}, status_code=503)
    import dataclasses
    return dataclasses.asdict(_voice_cfg)


@app.put("/api/voice/config")
async def api_voice_config_put(req: dict):
    """Update voice pipeline configuration at runtime."""
    if not VOICE_PIPELINE_AVAILABLE:
        return JSONResponse({"error": "Voice pipeline not available"}, status_code=503)
    try:
        voice_configure(req)
        return {"ok": True, "updated": list(req.keys())}
    except Exception:
        logger.exception("Failed to update voice pipeline configuration")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


@app.get("/api/voice/presets")
async def api_voice_presets_list():
    """List saved voice presets."""
    if not VOICE_PIPELINE_AVAILABLE:
        return JSONResponse({"error": "Voice pipeline not available"}, status_code=503)
    from voice.voice_config import VoiceConfig
    import os
    preset_dir = os.path.expanduser("~/.local/share/deltai/voice/presets")
    presets = []
    if os.path.isdir(preset_dir):
        for f in os.listdir(preset_dir):
            if f.endswith(".json"):
                presets.append(f.replace(".json", ""))
    return {"presets": presets}


@app.post("/api/voice/presets")
async def api_voice_preset_save(req: dict):
    """Save current config as a named preset."""
    if not VOICE_PIPELINE_AVAILABLE:
        return JSONResponse({"error": "Voice pipeline not available"}, status_code=503)
    name = req.get("name", "").strip()
    if not name:
        return JSONResponse({"error": "Preset name required"}, status_code=400)
    try:
        _voice_cfg.save_preset(name)
        return {"ok": True, "name": name}
    except Exception as e:
        return JSONResponse({"error": safe_errors.public_error_detail(e)}, status_code=500)


@app.get("/api/voice/status")
async def api_voice_pipeline_status():
    """Return voice pipeline health and component status."""
    if not VOICE_PIPELINE_AVAILABLE:
        return {"available": False, "error": "Voice pipeline module not loaded"}
    status = {
        "available": True,
        "tts": {"engine": "piper", "ready": False},
        "rvc": {"ready": False, "model_loaded": False},
        "effects": {"ready": True},
        "playback": {"ready": False},
    }
    try:
        from voice.tts_engine import PiperTTS
        status["tts"]["ready"] = PiperTTS is not None
    except Exception:
        pass
    try:
        from voice.voice_converter import VoiceConverter
        vc = VoiceConverter()
        status["rvc"]["model_loaded"] = vc.is_loaded
    except Exception:
        pass
    try:
        import sounddevice
        status["playback"]["ready"] = True
    except ImportError:
        status["playback"]["ready"] = False
    return status


@app.post("/api/voice/test")
async def api_voice_test():
    """Speak a test phrase and return timing breakdown."""
    if not VOICE_PIPELINE_AVAILABLE:
        return JSONResponse({"error": "Voice pipeline not available"}, status_code=503)
    import time
    timings = {}
    test_text = "All systems nominal. Voice pipeline test complete."
    try:
        from voice.tts_engine import PiperTTS
        from voice.post_processor import PostProcessor
        from voice.voice_config import DEFAULT_CONFIG
        tts = PiperTTS(DEFAULT_CONFIG)
        pp = PostProcessor(DEFAULT_CONFIG)
        t0 = time.time()
        audio = tts.synthesize(test_text)
        timings["tts_ms"] = round((time.time() - t0) * 1000)
        if audio is not None:
            t0 = time.time()
            processed = pp.process(audio)
            timings["effects_ms"] = round((time.time() - t0) * 1000)
            timings["audio_samples"] = len(processed)
            timings["duration_s"] = round(len(processed) / 22050, 2)
        else:
            timings["tts_ms"] = 0
            timings["note"] = "Piper TTS not installed — install piper-tts for local voice"
        return {"ok": True, "text": test_text, "timings": timings}
    except Exception as e:
        return {"ok": False, "error": safe_errors.public_error_detail(e), "timings": timings}


# ── VOICE TRAINING ENDPOINTS ────────────────────────────────────────────

@app.post("/api/voice/train/prepare")
async def api_voice_train_prepare():
    """Prepare training dataset from raw audio files."""
    if not VOICE_PIPELINE_AVAILABLE:
        return JSONResponse({"error": "Voice pipeline not available"}, status_code=503)
    try:
        from voice.train_rvc import prepare_dataset
        stats = await asyncio.to_thread(prepare_dataset)
        return {"ok": True, **stats}
    except Exception as e:
        return JSONResponse({"error": safe_errors.public_error_detail(e)}, status_code=500)


@app.post("/api/voice/train/start")
async def api_voice_train_start(req: dict = None):
    """Start RVC model training in background."""
    if not VOICE_PIPELINE_AVAILABLE:
        return JSONResponse({"error": "Voice pipeline not available"}, status_code=503)
    try:
        from voice.train_rvc import train
        output_name = (req or {}).get("model_name", None)
        train(output_name=output_name)
        return {"ok": True, "status": "training started"}
    except RuntimeError as e:
        return JSONResponse({"error": safe_errors.public_error_detail(e)}, status_code=409)
    except Exception as e:
        return JSONResponse({"error": safe_errors.public_error_detail(e)}, status_code=500)


@app.get("/api/voice/train/status")
async def api_voice_train_status():
    """Get RVC training progress."""
    if not VOICE_PIPELINE_AVAILABLE:
        return JSONResponse({"error": "Voice pipeline not available"}, status_code=503)
    from voice.train_rvc import get_training_state
    return get_training_state()


@app.post("/api/voice/train/stop")
async def api_voice_train_stop():
    """Abort RVC training."""
    if not VOICE_PIPELINE_AVAILABLE:
        return JSONResponse({"error": "Voice pipeline not available"}, status_code=503)
    from voice.train_rvc import stop_training
    stop_training()
    return {"ok": True, "status": "abort requested"}


@app.post("/api/voice/train/export")
async def api_voice_train_export(req: dict = None):
    """Export trained model for inference."""
    if not VOICE_PIPELINE_AVAILABLE:
        return JSONResponse({"error": "Voice pipeline not available"}, status_code=503)
    try:
        from voice.train_rvc import export_model
        output_name = (req or {}).get("model_name", None)
        result = await asyncio.to_thread(export_model, output_name=output_name)
        return {"ok": True, **result}
    except Exception as e:
        return JSONResponse({"error": safe_errors.public_error_detail(e)}, status_code=500)


# ── STATS ───────────────────────────────────────────────────────────────

@app.get("/stats")
def stats():
    result = {}
    result["cpu"] = {
        "name": platform.processor() or "Intel i7-12700K",
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True),
        "percent": psutil.cpu_percent(interval=None),
        "freq_mhz": round(psutil.cpu_freq().current) if psutil.cpu_freq() else 0,
    }
    ram = psutil.virtual_memory()
    result["ram"] = {
        "total_gb": round(ram.total / 1e9, 1),
        "used_gb": round(ram.used / 1e9, 1),
        "percent": ram.percent,
    }
    disk = psutil.disk_usage("/")
    result["disk"] = {
        "total_gb": round(disk.total / 1e9, 1),
        "used_gb": round(disk.used / 1e9, 1),
        "percent": disk.percent,
    }
    if GPU_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
            name = pynvml.nvmlDeviceGetName(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            try:
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000
            except Exception:
                power_limit = 170
            result["gpu"] = {
                "name": name if isinstance(name, str) else name.decode(),
                "vram_total_mb": round(mem.total / 1e6),
                "vram_used_mb": round(mem.used / 1e6),
                "vram_percent": round(mem.used / mem.total * 100, 1),
                "gpu_percent": util.gpu,
                "power_w": round(power, 1),
                "power_limit_w": round(power_limit, 1),
                "temp_c": temp,
            }
        except Exception as e:
            safe_errors.log_exception(logger, "stats GPU probe failed", e)
            result["gpu"] = {"error": safe_errors.public_error_detail(e)}
    else:
        result["gpu"] = {"error": "NVML unavailable"}
    try:
        resp = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        tags = resp.json()
        DELTAI_MODELS = {"deltai-qwen14b", "deltai-qwen3b", "deltai-nemo", "deltai-fallback", "deltai"}
        result["models"] = [m["name"] for m in tags.get("models", []) if m["name"].split(":")[0] in DELTAI_MODELS]
    except Exception:
        result["models"] = []
    if RAG_AVAILABLE:
        try:
            ms = get_memory_stats()
            result["memory_mb"] = ms["disk_mb"]
            result["memory_chunks"] = ms["total_chunks"]
            result["memory_files"] = ms["total_files"]
        except Exception:
            result["memory_mb"] = 0
            result["memory_chunks"] = 0
            result["memory_files"] = 0
    else:
        try:
            total = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, dn, fn in os.walk(os.path.expanduser(os.getenv("CHROMADB_PATH", "~/.local/share/deltai/chromadb")))
                for f in fn
            )
            result["memory_mb"] = round(total / 1e6, 1)
        except Exception:
            result["memory_mb"] = 0
    try:
        active_model, _, _ = _pick_local_model()
        result["model"] = active_model
    except Exception:
        result["model"] = DELTAI_MODEL
    result["platform"] = platform.system() + " " + platform.release()
    result["rag_available"] = RAG_AVAILABLE
    result["cloud_available"] = is_cloud_available_sync()
    result["watcher_active"] = watcher_running() if RAG_AVAILABLE else False
    # Backup status — minimal, invisible infrastructure
    backup_enabled = os.getenv("BACKUP_ENABLED", "true").lower() in ("true", "1", "yes")
    all_healthy = all(s.get("healthy", False) for s in _backup_health_status.values()) if _backup_health_status else True
    result["backup"] = {"enabled": backup_enabled, "healthy": all_healthy}
    result["budget"] = get_budget_status()
    return result


# ── MODELFILE ───────────────────────────────────────────────────────────

MODELFILE_PATH = r"~/deltai/modelfiles\deltai.modelfile"
MODULES_DIR = os.path.join(_HERE, '..', 'modelfiles')
MODULE_FILES = {
    "modelfile": MODELFILE_PATH,
    "protocols": os.path.join(MODULES_DIR, "protocols.md"),
    "personality": os.path.join(MODULES_DIR, "personality.md"),
    "pilot": os.path.join(MODULES_DIR, "pilot.md"),
}
_ALLOWED_MODULE_PATHS = frozenset(MODULE_FILES.values())


@app.get("/modelfile")
def get_modelfile():
    try:
        with open(MODELFILE_PATH, "r", encoding="utf-8") as f:
            return PlainTextResponse(f.read())
    except Exception as e:
        safe_errors.log_exception(logger, "get_modelfile failed", e)
        return PlainTextResponse("# Error reading modelfile", status_code=500)

class ModelfileUpdate(BaseModel):
    content: str

@app.post("/modelfile")
def save_modelfile(update: ModelfileUpdate):
    try:
        with open(MODELFILE_PATH, "w", encoding="utf-8") as f:
            f.write(update.content)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": safe_errors.public_error_detail(e)}

@app.get("/module/{name}")
def get_module(name: str):
    path = MODULE_FILES.get(name)
    if not path or path not in _ALLOWED_MODULE_PATHS:
        return PlainTextResponse(f"# Unknown module: {name}", status_code=404)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return PlainTextResponse(f.read())
    except Exception as e:
        safe_errors.log_exception(logger, f"get_module {name} failed", e)
        return PlainTextResponse(f"# Error reading {name}", status_code=500)

@app.post("/module/{name}")
def save_module(name: str, update: ModelfileUpdate):
    path = MODULE_FILES.get(name)
    if not path or path not in _ALLOWED_MODULE_PATHS:
        return {"ok": False, "error": f"Unknown module: {name}"}
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(update.content)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": safe_errors.public_error_detail(e)}
