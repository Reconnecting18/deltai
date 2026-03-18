from contextlib import asynccontextmanager
from collections import deque
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse, PlainTextResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
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

from tools.definitions import TOOLS
from tools.executor import execute_tool
from router import (route, is_cloud_available, is_cloud_available_sync,
                    get_gpu_utilization, get_vram_free_mb, classify_complexity,
                    is_sim_running, get_budget_status, record_cloud_usage,
                    _pick_local_model, get_backup_model, check_model_health,
                    check_model_exists, init_budget_from_db,
                    is_session_active, activate_session, deactivate_session,
                    activate_session_from_ingest, get_session_status,
                    classify_telemetry_category)
from anthropic_client import stream_chat as anthropic_stream
from persistence import (init_db, load_history, save_history_pair,
                         clear_history as db_clear_history, trim_history,
                         export_session_history)

load_dotenv()
psutil.cpu_percent(interval=0.1)

logger = logging.getLogger("e3n")

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

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
E3N_MODEL  = os.getenv("E3N_MODEL", "e3n")
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
    """Check if E3N is idle (no active chat, 30s breathing room after last chat)."""
    if _last_chat_start > _last_chat_end:
        return False  # chat in progress
    if _time.time() - _last_chat_end < 30:
        return False  # recently finished
    return True


# ── TELEMETRY PROMPT TEMPLATES ───────────────────────────────────────

_COACH_TEMPLATE = """You are in COACHING MODE. The operator is asking about driving performance.
Structure your response as:
1. OBSERVATION — what the data shows (specific numbers)
2. IMPACT — how it affects lap time (quantify in tenths/hundredths)
3. ACTION — one concrete change to try next lap
Compare to the driver's own best, not theoretical limits. Be surgical — data over feelings."""

_ENGINEER_TEMPLATE = """You are in RACE ENGINEER MODE. The operator needs a strategic decision.
Structure your response as:
1. RECOMMENDATION — lead with the call (pit/stay/save fuel/etc.)
2. NUMBERS — lap to pit, compound choice, fuel target, time delta
3. RISK — what could go wrong, and the fallback plan
Never hedge a pit call. Give a clear recommendation with confidence. Include specific numbers."""


def build_telemetry_prompt(query_category: str) -> str | None:
    """Return the coaching or engineering prompt template for telemetry queries."""
    if query_category == "telemetry_coaching":
        return _COACH_TEMPLATE
    elif query_category in ("telemetry_strategy", "telemetry_debrief"):
        return _ENGINEER_TEMPLATE
    return None


# ── CONVERSATION HISTORY ─────────────────────────────────────────────
HISTORY_MAX_TURNS = int(os.getenv("CONVERSATION_HISTORY_MAX", "10"))
_conversation_history: list[dict] = []
_current_session_id: str | None = None  # tracks active session for history tagging


def _append_to_history(user_message: str, assistant_response: str,
                       query_category: str = "general", rag_context: str = ""):
    """Store a clean user-assistant exchange. Skip if either is empty."""
    global _last_chat_end
    _last_chat_end = _time.time()
    if not user_message.strip() or not assistant_response.strip():
        return
    _conversation_history.append({"role": "user", "content": user_message})
    _conversation_history.append({"role": "assistant", "content": assistant_response})
    max_msgs = HISTORY_MAX_TURNS * 2
    if len(_conversation_history) > max_msgs:
        del _conversation_history[:-max_msgs]
    # Persist to SQLite (with session tagging)
    try:
        save_history_pair(user_message, assistant_response, session_id=_current_session_id)
        trim_history(HISTORY_MAX_TURNS)
    except Exception as e:
        logger.warning(f"Failed to persist history: {e}")
    # Auto-capture good exchanges for training data
    if TRAINING_AVAILABLE:
        try:
            # Route racing exchanges to dedicated dataset
            if query_category.startswith("telemetry_"):
                auto_capture("e3n-racing", user_message, assistant_response,
                             category=query_category, rag_context=rag_context)
            else:
                auto_capture("e3n-auto", user_message, assistant_response)
        except Exception:
            pass  # never break chat for training


def _get_history() -> list[dict]:
    """Return a copy of conversation history for injection into message arrays."""
    return list(_conversation_history)


async def _try_ollama_inference(client: httpx.AsyncClient, model: str,
                                messages: list, tools: list | None = None) -> tuple[dict | None, str | None]:
    """
    Attempt a single Ollama inference call with circuit breaker protection.
    Returns (data, None) on success or (None, error_string) on failure.
    """
    # Circuit breaker: don't hammer Ollama when it's down
    if not _cb_check():
        return None, f"Circuit breaker open — Ollama unreachable (backoff {_circuit_breaker['backoff_sec']}s)"

    payload = {"model": model, "messages": messages, "stream": False}
    if tools:
        payload["tools"] = tools
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
        return None, str(e)


async def _inference_with_emergency_fallback(
    client: httpx.AsyncClient, model: str, messages: list,
    tools: list | None, backup_model: str | None,
) -> tuple[dict | None, str | None, str, bool]:
    """
    Try primary model with retries, then emergency backup as last resort.

    Returns (data, error, used_model, is_emergency).
    """
    # ── Attempt primary (with retries) ──
    last_error = None
    for attempt in range(1 + BACKUP_MAX_RETRIES):
        data, err = await _try_ollama_inference(client, model, messages, tools)
        if data is not None:
            return data, None, model, False
        last_error = err
        if attempt < BACKUP_MAX_RETRIES:
            logger.warning(f"Primary model {model} failed (attempt {attempt + 1}): {err} — retrying in 3s")
            await asyncio.sleep(3)

    # ── Primary exhausted — engage emergency backup ──
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
        for key in ("E3N_BACKUP_STRONG_MODEL", "E3N_BACKUP_MODEL"):
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
}
_RESOURCE_CHECK_INTERVAL = 30    # seconds between resource checks
_VRAM_CRITICAL_MB = 1500        # below this, aggressively free VRAM
_VRAM_WARN_MB = 3000            # below this, start monitoring closely
_MAX_WATCHER_RESTARTS = 5       # don't restart watcher more than this
_OLLAMA_FAILURE_THRESHOLD = 3   # consecutive failures before auto-restart attempt
_RESOURCE_ACTION_COOLDOWN = 60  # seconds between automatic VRAM actions


def _log_resource_action(action: str):
    """Record an automatic resource management action."""
    entry = {"action": action, "ts": _time.time()}
    _resource_state["actions_taken"].append(entry)
    if len(_resource_state["actions_taken"]) > 50:
        _resource_state["actions_taken"] = _resource_state["actions_taken"][-50:]
    logger.info(f"Resource manager: {action}")


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

            # ── 1. VRAM MANAGEMENT ──
            vram_free = get_vram_free_mb()
            sim_active = is_sim_running()

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
                        sim_model = os.getenv("E3N_SIM_MODEL", os.getenv("E3N_MODEL", "e3n-qwen3b"))
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
                    strong_model = os.getenv("E3N_STRONG_MODEL", "e3n-qwen14b")
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
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
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
            if RAG_AVAILABLE and is_session_active():
                try:
                    cleanup_expired()
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
_SELF_HEAL_MODEL = os.getenv("E3N_MODEL", "e3n-qwen3b")

_SELF_HEAL_SYSTEM_PROMPT = """You are E3N's internal diagnostics AI. You receive a system diagnostics report and must decide if any repairs are needed.

RULES:
- Only suggest repairs if there are actual issues (DOWN, ERROR, STOPPED, WARNING, DEGRADED)
- Available repairs: restart_watcher, clear_vram, reindex_knowledge, check_ollama
- If no issues found, respond with exactly: NO_ACTION
- If repair needed, respond with exactly: REPAIR:<repair_name>
- Only suggest ONE repair per cycle (most critical first)
- Never suggest clear_vram during an active racing session
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

            # Skip during active racing sessions (GPU protection)
            if is_session_active():
                logger.debug("Self-heal: skipping (racing session active)")
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    if RAG_AVAILABLE:
        try:
            results = ingest_all()
            ingested = [r for r in results if r["status"] == "ok"]
            if ingested:
                logger.info(f"Initial ingestion: {len(ingested)} file(s)")
            start_watcher()
            logger.info("File watcher started")
        except Exception as e:
            logger.error(f"RAG startup failed: {e}")

    # ── Persistence ──
    try:
        init_db()
        loaded = load_history(HISTORY_MAX_TURNS)
        _conversation_history.extend(loaded)
        logger.info(f"Loaded {len(loaded) // 2} conversation turns from DB")
        init_budget_from_db()
    except Exception as e:
        logger.error(f"Persistence startup failed: {e}")

    cloud = await is_cloud_available()
    logger.info(f"Cloud available: {cloud}")
    logger.info(f"GPU utilization: {get_gpu_utilization()}%")

    # ── Model existence checks ──
    primary_models = [
        ("PRIMARY", os.getenv("E3N_STRONG_MODEL", "e3n-qwen14b")),
        ("PRIMARY", os.getenv("E3N_MODEL", "e3n-qwen3b")),
    ]
    backup_models = [
        ("BACKUP", os.getenv("E3N_BACKUP_STRONG_MODEL", "e3n-nemo")),
        ("BACKUP", os.getenv("E3N_BACKUP_MODEL", "e3n")),
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

    # ── Start background tasks ──
    health_task = asyncio.create_task(_backup_health_loop())
    resource_task = asyncio.create_task(_resource_manager_loop())
    self_heal_task = asyncio.create_task(_ai_self_heal_loop())

    yield

    # ── Shutdown ──
    for task in (health_task, resource_task, self_heal_task):
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

app = FastAPI(title="E3N", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=os.path.join(_HERE, "static")), name="static")


# ── TEXT-AS-TOOL FALLBACK PARSER ────────────────────────────────────────

def _sanitize_python_json(text: str) -> str:
    """Replace Python literals (True/False/None) with JSON equivalents."""
    text = re.sub(r'(?<=[\s:,\[])True(?=[\s,}\]])', 'true', text)
    text = re.sub(r'(?<=[\s:,\[])False(?=[\s,}\]])', 'false', text)
    text = re.sub(r'(?<=[\s:,\[])None(?=[\s,}\]])', 'null', text)
    return text

def _fix_windows_paths(text: str) -> str:
    """Fix invalid JSON escape sequences from Windows paths (e.g., C:\\e3n → C:\\\\e3n)."""
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
    # Greetings
    "hey": "Operational.",
    "hi": "Operational.",
    "hello": "Operational.",
    "yo": "Operational.",
    "hey e3n": "Operational.",
    "hi e3n": "Operational.",
    "hello e3n": "Operational.",
    "hey e3": "Operational.",
    "hi e3": "Operational.",
    # What's up
    "sup": "Standing by.",
    "whats up": "Standing by.",
    "what's up": "Standing by.",
    "wassup": "Standing by.",
    # Morning / evening
    "morning": "Online. Morning, Ethan.",
    "good morning": "Online. Morning, Ethan.",
    "morning e3n": "Online. Morning, Ethan.",
    "morning e3": "Online. Morning, Ethan.",
    "gm": "Online. Morning, Ethan.",
    "evening": "Online. Evening, Ethan.",
    "good evening": "Online. Evening, Ethan.",
    # Night / goodbye
    "night": "Powering down comms. Night, Ethan.",
    "goodnight": "Powering down comms. Night, Ethan.",
    "good night": "Powering down comms. Night, Ethan.",
    "gn": "Powering down comms. Night, Ethan.",
    "bye": "Copy. Out.",
    "later": "Copy. Out.",
    "peace": "Copy. Out.",
    "see ya": "Copy. Out.",
    "goodbye": "Copy. Out.",
    # Status checks
    "you there?": "Online.",
    "you there": "Online.",
    "online?": "Online.",
    "awake?": "Online.",
    "you up?": "Online.",
    "e3n?": "Online.",
    "e3?": "Online.",
}


def _check_greeting(message: str) -> str | None:
    """
    Check if the message is a simple greeting/farewell.
    Returns a canned E3N response or None if not a greeting.
    """
    text = message.strip().lower().rstrip("!.,")
    return _GREETING_MAP.get(text)


# ── RAG CONTEXT BUILDER ────────────────────────────────────────────────

def build_rag_context(user_message: str, source_filter: str = None,
                      max_age_sec: float = None) -> str:
    if not RAG_AVAILABLE:
        return ""
    # Cleanup expired entries during active sessions
    if is_session_active():
        try:
            cleanup_expired()
        except Exception:
            pass
    try:
        matches = query_knowledge(user_message, n_results=3, threshold=0.75,
                                  source_filter=source_filter, max_age_sec=max_age_sec)
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
        logger.info(f"Greeting short-circuit: '{req.message}' → '{greeting_response}'")
        return StreamingResponse(stream_greeting(), media_type="application/x-ndjson")

    # ── SESSION EVENT ──────────────────────────────────────────
    query_cat = decision.query_category

    # ── TELEMETRY LOOKUP SHORT-CIRCUIT ─────────────────────────
    # Simple data queries during active session — answer from RAG, no model needed
    if query_cat == "telemetry_lookup" and RAG_AVAILABLE:
        lookup_context = build_rag_context(req.message, source_filter=None, max_age_sec=120)
        if lookup_context:
            async def stream_lookup():
                yield json.dumps({"t": "route", **decision.to_dict()}) + "\n"
                chunk_count = lookup_context.count("[Source:")
                yield json.dumps({"t": "rag", "n": chunk_count}) + "\n"
                # Extract clean data from RAG context (strip metadata headers)
                clean = lookup_context.replace("[KNOWLEDGE CONTEXT — from your ingested documents]", "").replace("[END CONTEXT]", "")
                clean = re.sub(r'\[Source:.*?\|.*?\]\n?', '', clean).strip()
                yield json.dumps({"t": "text", "c": clean}) + "\n"
                _append_to_history(req.message, clean, query_category=query_cat, rag_context=lookup_context)
                yield json.dumps({"t": "done", "turns": len(_conversation_history) // 2}) + "\n"
            logger.info(f"Telemetry lookup short-circuit: '{req.message}'")
            return StreamingResponse(stream_lookup(), media_type="application/x-ndjson")

    # ── RAG CONTEXT ─────────────────────────────────────────
    # During active sessions, prefer recent telemetry data
    if is_session_active() and query_cat.startswith("telemetry_"):
        rag_context = build_rag_context(req.message, max_age_sec=300)
    else:
        rag_context = build_rag_context(req.message)

    # ── TELEMETRY PROMPT INJECTION ──────────────────────────
    telemetry_prompt = build_telemetry_prompt(query_cat)

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
                    history=_get_history(), telemetry_mode=telemetry_prompt,
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
                            _append_to_history(req.message, full_response, query_category=query_cat, rag_context=rag_context)
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
                            client, local_model, local_messages, TOOLS, backup
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
                    history=_get_history(), telemetry_mode=telemetry_prompt,
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
                            _append_to_history(req.message, full_response, query_category=query_cat, rag_context=rag_context)
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
                split_mode=True, telemetry_mode=telemetry_prompt,
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
                        _append_to_history(req.message, full_response, query_category=query_cat, rag_context=rag_context)
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

            async for line in anthropic_stream(
                message=req.message,
                model=decision.model,
                rag_context=rag_context,
                tools=TOOLS,
                execute_tool_fn=execute_tool,
                history=_get_history(),
                telemetry_mode=telemetry_prompt,
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
                        _append_to_history(req.message, full_response, query_category=query_cat, rag_context=rag_context)
                        yield json.dumps({"t": "done", "turns": len(_conversation_history) // 2}) + "\n"
                        continue
                except (json.JSONDecodeError, ValueError):
                    pass
                yield line

        return StreamingResponse(stream_cloud(), media_type="application/x-ndjson")

    # ── OLLAMA PATH (with tool calling) ─────────────────────
    # Inject telemetry prompt as system message for local models
    telemetry_prefix = ""
    if telemetry_prompt:
        telemetry_prefix = f"[SYSTEM NOTE]\n{telemetry_prompt}\n[END NOTE]\n\n"

    if rag_context:
        user_content = f"{telemetry_prefix}{rag_context}\n{req.message}"
    else:
        user_content = f"{telemetry_prefix}{req.message}" if telemetry_prefix else req.message

    messages = _get_history() + [{"role": "user", "content": user_content}]

    async def stream_local():
        nonlocal messages

        yield json.dumps({"t": "route", **decision.to_dict()}) + "\n"

        if rag_context:
            chunk_count = rag_context.count("[Source:")
            yield json.dumps({"t": "rag", "n": chunk_count}) + "\n"

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
                    client, active_model, messages, TOOLS, backup if not emergency_active else None
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
                    if "timed out" in str(err).lower():
                        friendly = "Model took too long to respond. Try a shorter question or check if Ollama is running."
                    elif "connect" in str(err).lower() or "refused" in str(err).lower():
                        friendly = "Cannot reach Ollama. Make sure it's running (ollama serve)."
                    elif "all models failed" in str(err).lower():
                        friendly = "All models are offline. Check Ollama and model availability."
                    else:
                        friendly = f"Backend error: {err}"
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
                    _append_to_history(req.message, content, query_category=query_cat, rag_context=rag_context)
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
                        result = f"ERROR: Tool execution crashed: {tool_err}"
                    # If tool errored, retry once with sanitized args
                    if result.startswith("ERROR:") and name in ("run_powershell", "read_file", "write_file"):
                        yield json.dumps({"t": "retry", "n": name, "c": "Retrying with adjusted parameters..."}) + "\n"
                        sanitized_args = dict(args)
                        if "path" in sanitized_args:
                            sanitized_args["path"] = os.path.normpath(sanitized_args["path"])
                        if "timeout" in sanitized_args and name == "run_powershell":
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
                client, active_model, messages, None, backup if not emergency_active else None
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
                _append_to_history(req.message, content, query_category=query_cat, rag_context=rag_context)
            else:
                yield json.dumps({"t": "error", "c": str(err)}) + "\n"
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
        "local_model": os.getenv("E3N_MODEL", "e3n"),
        "strong_model": os.getenv("E3N_STRONG_MODEL", ""),
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
        return {"error": str(e)}

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
        return {"error": str(e)}

@app.get("/memory/files")
def memory_files_endpoint():
    """Get detailed per-file info from ChromaDB."""
    if not RAG_AVAILABLE:
        return {"error": "RAG not available", "files": []}
    try:
        files = get_file_details()
        return {"files": files}
    except Exception as e:
        return {"error": str(e), "files": []}

@app.delete("/memory/files/{filepath:path}")
def memory_delete_file(filepath: str):
    """Remove a specific file's chunks from ChromaDB."""
    if not RAG_AVAILABLE:
        return {"error": "RAG not available"}
    try:
        full_path = os.path.normpath(os.path.join(KNOWLEDGE_PATH, filepath))
        if not full_path.startswith(os.path.normpath(KNOWLEDGE_PATH)):
            return {"status": "error", "reason": "Invalid path — outside knowledge directory"}
        result = remove_file(full_path)
        return result
    except Exception as e:
        return {"status": "error", "reason": str(e)}


# ── INGEST CONNECTOR ──────────────────────────────────────────────────
# External services push structured context into E3N's RAG memory.

class IngestRequest(BaseModel):
    source: str               # identifies the sender (e.g., "lmu-telemetry")
    context: str              # human-readable content to embed
    ttl: int = 0              # seconds until auto-expiry (0 = permanent)
    tags: list[str] = []      # optional filtering tags


@app.post("/ingest")
async def ingest_endpoint(req: IngestRequest):
    """
    Ingest structured context from an external service into ChromaDB.
    This is E3N's connector — any service can push context here.
    """
    if not RAG_AVAILABLE:
        return {"error": "RAG not available"}
    if not req.source or not req.source.strip():
        return {"status": "error", "reason": "source is required"}
    if not req.context or not req.context.strip():
        return {"status": "error", "reason": "context is required"}
    try:
        # Auto-activate session from telemetry source
        global _current_session_id
        was_active = is_session_active()
        activate_session_from_ingest(req.source.strip())
        # Auto-set session_id when ingest triggers session activation
        if not was_active and is_session_active() and not _current_session_id:
            _current_session_id = f"auto_{int(_time.time())}"
        result = await asyncio.to_thread(
            ingest_context,
            source=req.source.strip(),
            context=req.context.strip(),
            ttl=req.ttl,
            tags=req.tags,
        )
        # Forward alerts to WebSocket clients
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
        return {"status": "error", "reason": str(e)}


class IngestBatchRequest(BaseModel):
    items: list[dict]  # each: {"source": str, "context": str, "ttl": int, "tags": list[str]}


@app.post("/ingest/batch")
async def ingest_batch_endpoint(req: IngestBatchRequest):
    """Batch ingest multiple context items. Max 100 per batch."""
    if not RAG_AVAILABLE:
        return {"error": "RAG not available"}
    try:
        # Auto-activate session from any telemetry sources in batch
        for item in req.items[:5]:  # check first 5 for pattern
            src = item.get("source", "")
            if src:
                activate_session_from_ingest(src)
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
        return {"status": "error", "reason": str(e)}


@app.post("/ingest/cleanup")
def ingest_cleanup():
    """Manually trigger TTL cleanup of expired ingested entries."""
    if not RAG_AVAILABLE:
        return {"error": "RAG not available"}
    try:
        return cleanup_expired()
    except Exception as e:
        return {"status": "error", "reason": str(e)}


# ── SESSION ENDPOINTS ────────────────────────────────────────────────

@app.post("/session/start")
def session_start():
    """Manually activate racing session mode (GPU protection)."""
    global _current_session_id
    _current_session_id = f"session_{int(_time.time())}"
    activate_session(manual=True)
    return {"ok": True, "session_id": _current_session_id, **get_session_status()}


@app.post("/session/end")
def session_end():
    """Deactivate racing session mode and export session history."""
    global _current_session_id
    result = {"ok": True}
    # Export session history for training
    if _current_session_id:
        try:
            export_path = os.path.join(r"C:\e3n\data\training\sessions", f"{_current_session_id}.jsonl")
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            export_result = export_session_history(_current_session_id, export_path)
            result["export"] = export_result
        except Exception as e:
            result["export_error"] = str(e)
    _current_session_id = None
    deactivate_session()
    result.update(get_session_status())
    return result


@app.get("/session/status")
def session_status():
    """Get current session state."""
    status = get_session_status()
    status["session_id"] = _current_session_id
    return status


# ── WEBSOCKET ALERTS ─────────────────────────────────────────────────

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket endpoint for proactive racing alerts."""
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
        "session_active": is_session_active(),
        "circuit_breaker": _circuit_breaker["state"],
        "recent_actions": heal_actions[-10:],
    }


# ── RESOURCE MANAGEMENT STATUS ────────────────────────────────────────

@app.get("/resources/status")
def resource_status():
    """Resource self-manager status — VRAM, circuit breaker, auto-recovery."""
    return {
        "resource_manager": {
            "vram_free_mb": get_vram_free_mb(),
            "vram_warnings": _resource_state["vram_warnings"],
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
    """Returns status of all E3N subsystems for the header health monitor."""
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
    strong = os.getenv("E3N_STRONG_MODEL", "e3n-qwen14b")
    default = os.getenv("E3N_MODEL", "e3n-qwen3b")
    return {
        "enabled": backup_enabled,
        "mapping": {
            strong: os.getenv("E3N_BACKUP_STRONG_MODEL", "e3n-nemo"),
            default: os.getenv("E3N_BACKUP_MODEL", "e3n"),
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
    input: str
    output: str
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
    return add_example(name, req.input, req.output, req.category)

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
    base_model: str = "e3n-qwen3b"
    output_model: str | None = None
    mode: str = "auto"  # "lora", "fewshot", or "auto"


@app.post("/training/start")
def training_start(req: TrainingStart):
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    return start_training(
        dataset_name=req.dataset,
        base_model=req.base_model,
        output_model=req.output_model,
        mode=req.mode,
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


# ── VOICE ENDPOINTS ──────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str
    voice: str = None
    rate: str = None


@app.get("/voice/status")
def voice_status():
    """Voice subsystem status."""
    if not VOICE_AVAILABLE:
        return {"enabled": False, "error": "Voice module not loaded",
                "stt": {"status": "unavailable"}, "tts": {"status": "unavailable"}}
    return get_voice_status()


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
    result = transcribe_audio(body)
    return result


@app.post("/voice/tts")
async def voice_tts(req: TTSRequest):
    """Convert text to speech audio. Returns audio bytes."""
    if not VOICE_AVAILABLE:
        return JSONResponse({"error": "Voice module not available"}, status_code=503)
    result = await synthesize_speech(req.text, voice=req.voice, rate=req.rate)
    if "error" in result:
        return JSONResponse({"error": result["error"]}, status_code=500)
    return Response(
        content=result["audio"],
        media_type=f"audio/{result['format']}",
        headers={
            "Content-Disposition": f"inline; filename=e3n_speech.{result['format']}",
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
    stt_result = transcribe_audio(body)
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
        user_content = f"{rag_context}\n{user_text}" if rag_context else user_text
        msg_list = _get_history() + [{"role": "user", "content": user_content}]

        async with httpx.AsyncClient(timeout=120) as client:
            data, err, used_model, is_emergency = await _inference_with_emergency_fallback(
                client, decision.model, msg_list, TOOLS, decision.backup_model
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
    disk = psutil.disk_usage("C:\\")
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
            result["gpu"] = {"error": str(e)}
    else:
        result["gpu"] = {"error": "NVML unavailable"}
    try:
        resp = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        tags = resp.json()
        E3N_MODELS = {"e3n-qwen14b", "e3n-qwen3b", "e3n-nemo", "e3n"}
        result["models"] = [m["name"] for m in tags.get("models", []) if m["name"].split(":")[0] in E3N_MODELS]
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
                for dp, dn, fn in os.walk(os.getenv("CHROMADB_PATH", "C:\\e3n\\data\\chromadb"))
                for f in fn
            )
            result["memory_mb"] = round(total / 1e6, 1)
        except Exception:
            result["memory_mb"] = 0
    try:
        active_model, _, _ = _pick_local_model()
        result["model"] = active_model
    except Exception:
        result["model"] = E3N_MODEL
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

MODELFILE_PATH = r"C:\e3n\modelfiles\E3N.modelfile"
MODULES_DIR = os.path.join(_HERE, '..', 'modelfiles')
MODULE_FILES = {
    "modelfile": MODELFILE_PATH,
    "protocols": os.path.join(MODULES_DIR, "protocols.md"),
    "personality": os.path.join(MODULES_DIR, "personality.md"),
    "pilot": os.path.join(MODULES_DIR, "pilot.md"),
}

@app.get("/modelfile")
def get_modelfile():
    try:
        with open(MODELFILE_PATH, "r", encoding="utf-8") as f:
            return PlainTextResponse(f.read())
    except Exception as e:
        return PlainTextResponse(f"# Error: {e}")

class ModelfileUpdate(BaseModel):
    content: str

@app.post("/modelfile")
def save_modelfile(update: ModelfileUpdate):
    try:
        with open(MODELFILE_PATH, "w", encoding="utf-8") as f:
            f.write(update.content)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/module/{name}")
def get_module(name: str):
    path = MODULE_FILES.get(name)
    if not path:
        return PlainTextResponse(f"# Unknown module: {name}", status_code=404)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return PlainTextResponse(f.read())
    except Exception as e:
        return PlainTextResponse(f"# Error reading {name}: {e}")

@app.post("/module/{name}")
def save_module(name: str, update: ModelfileUpdate):
    path = MODULE_FILES.get(name)
    if not path:
        return {"ok": False, "error": f"Unknown module: {name}"}
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(update.content)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}
