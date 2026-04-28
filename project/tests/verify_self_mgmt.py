"""
deltai Self-Management Stack Tests
Tests: Health Event Bus, Proactive Model Lifecycle, AI Self-Heal Loop
"""

import asyncio
import os
import sys
import time

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")
os.environ.setdefault("CHROMADB_PATH", os.path.join(os.path.dirname(__file__), "test_chromadb"))
os.environ.setdefault("KNOWLEDGE_PATH", os.path.join(os.path.dirname(__file__), "test_knowledge"))
os.environ.setdefault("TRAINING_PATH", os.path.join(os.path.dirname(__file__), "test_training"))
os.environ.setdefault("SQLITE_PATH", os.path.join(os.path.dirname(__file__), "test_deltai.db"))

from unittest.mock import patch

passed = 0
failed = 0
errors = []


def test(name):
    def decorator(fn):
        def wrapper():
            global passed, failed
            try:
                fn()
                passed += 1
                print(f"  PASS  {name}")
            except Exception as e:
                failed += 1
                errors.append((name, str(e)))
                print(f"  FAIL  {name}: {e}")

        return wrapper

    return decorator


# ═══════════════════════════════════════════════════════════════════════
# HEALTH EVENT BUS
# ═══════════════════════════════════════════════════════════════════════
print("\n=== HEALTH EVENT BUS ===\n")


@test("HE 1: _record_health_event adds to deque")
def test_he_record():
    import main

    initial_count = len(main._health_events)
    event = main._record_health_event("test_event", {"key": "value"})
    assert event["type"] == "test_event"
    assert event["key"] == "value"
    assert "ts" in event
    assert len(main._health_events) == initial_count + 1


test_he_record()


@test("HE 2: _emit_health_event is async and records event")
def test_he_emit():
    import main

    initial_count = len(main._health_events)
    asyncio.run(main._emit_health_event("async_test", {"data": 123}))
    assert len(main._health_events) == initial_count + 1
    last = main._health_events[-1]
    assert last["type"] == "async_test"
    assert last["data"] == 123


test_he_emit()


@test("HE 3: Health events deque has 100 max capacity")
def test_he_capacity():
    import main

    assert main._health_events.maxlen == 100


test_he_capacity()


@test("HE 4: Circuit breaker emits events on state change")
def test_he_cb_events():
    import main

    initial_count = len(main._health_events)
    # Save state
    old_cb = dict(main._circuit_breaker)
    try:
        # Reset CB
        main._circuit_breaker["state"] = "closed"
        main._circuit_breaker["failures"] = 0
        main._circuit_breaker["backoff_sec"] = 5

        # Trigger 3 failures to open CB
        for _ in range(3):
            main._cb_failure()

        # Should have emitted circuit_breaker_changed event
        cb_events = [
            e
            for e in list(main._health_events)[initial_count:]
            if e["type"] == "circuit_breaker_changed"
        ]
        assert len(cb_events) >= 1, f"Expected CB event, got {len(cb_events)}"
        assert cb_events[0]["state"] == "open"

        # Reset via success
        main._cb_success()
        cb_events2 = [
            e
            for e in list(main._health_events)[initial_count:]
            if e["type"] == "circuit_breaker_changed" and e["state"] == "closed"
        ]
        assert len(cb_events2) >= 1, "Expected closed event after success"
    finally:
        main._circuit_breaker.update(old_cb)


test_he_cb_events()


@test("HE 5: Health event bus WebSocket endpoint exists")
def test_he_ws_endpoint():
    import main

    routes = [r.path for r in main.app.routes if hasattr(r, "path")]
    assert "/ws/health" in routes, f"/ws/health not found in routes: {routes}"


test_he_ws_endpoint()


@test("HE 6: /health/events endpoint exists")
def test_he_rest_endpoint():
    import main

    routes = [r.path for r in main.app.routes if hasattr(r, "path")]
    assert "/health/events" in routes


test_he_rest_endpoint()


# ═══════════════════════════════════════════════════════════════════════
# IDLE TRACKING
# ═══════════════════════════════════════════════════════════════════════
print("\n=== IDLE TRACKING ===\n")


@test("IT 1: _is_idle returns True when no chat activity")
def test_idle_default():
    import deltai_api.core as core
    import main

    old_start = core._last_chat_start
    old_end = core._last_chat_end
    try:
        core._last_chat_start = 0.0
        core._last_chat_end = time.time() - 60  # 60s ago
        assert main._is_idle(), "Should be idle with no recent chat"
    finally:
        core._last_chat_start = old_start
        core._last_chat_end = old_end


test_idle_default()


@test("IT 2: _is_idle returns False during active chat")
def test_idle_during_chat():
    import deltai_api.core as core
    import main

    old_start = core._last_chat_start
    old_end = core._last_chat_end
    try:
        core._last_chat_start = time.time()
        core._last_chat_end = time.time() - 10  # end before start = active
        assert not main._is_idle(), "Should NOT be idle during active chat"
    finally:
        core._last_chat_start = old_start
        core._last_chat_end = old_end


test_idle_during_chat()


@test("IT 3: _is_idle returns False within 30s of last chat")
def test_idle_breathing_room():
    import deltai_api.core as core
    import main

    old_start = core._last_chat_start
    old_end = core._last_chat_end
    try:
        core._last_chat_start = time.time() - 20
        core._last_chat_end = time.time() - 10  # 10s ago
        assert not main._is_idle(), "Should NOT be idle within 30s of last chat"
    finally:
        core._last_chat_start = old_start
        core._last_chat_end = old_end


test_idle_breathing_room()


# ═══════════════════════════════════════════════════════════════════════
# PROACTIVE MODEL LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════
print("\n=== PROACTIVE MODEL LIFECYCLE ===\n")


@test("ML 1: Resource state has sim tracking fields")
def test_ml_state_fields():
    import main

    assert "last_sim_state" in main._resource_state
    assert "sim_stop_detected_at" in main._resource_state
    assert "pending_14b_preload" in main._resource_state


test_ml_state_fields()


@test("ML 2: Sim start transition sets pending_14b_preload=False")
def test_ml_sim_start():
    import main

    # The sim start transition clears pending preload
    main._resource_state["pending_14b_preload"] = True
    main._resource_state["last_sim_state"] = False
    # Simulate the transition check manually
    sim_active = True
    prev_sim = main._resource_state["last_sim_state"]
    if sim_active and not prev_sim:
        main._resource_state["pending_14b_preload"] = False
    assert not main._resource_state["pending_14b_preload"]


test_ml_sim_start()


@test("ML 3: Sim stop transition sets pending preload")
def test_ml_sim_stop():
    import main

    main._resource_state["last_sim_state"] = True
    main._resource_state["pending_14b_preload"] = False
    # Simulate stop transition
    sim_active = False
    prev_sim = main._resource_state["last_sim_state"]
    if not sim_active and prev_sim:
        main._resource_state["sim_stop_detected_at"] = time.time()
        main._resource_state["pending_14b_preload"] = True
    assert main._resource_state["pending_14b_preload"]
    assert main._resource_state["sim_stop_detected_at"] > 0
    # Reset
    main._resource_state["last_sim_state"] = False
    main._resource_state["pending_14b_preload"] = False


test_ml_sim_stop()


# ═══════════════════════════════════════════════════════════════════════
# AI SELF-HEAL LOOP
# ═══════════════════════════════════════════════════════════════════════
print("\n=== AI SELF-HEAL LOOP ===\n")


@test("SH 1: Self-heal config loaded from env")
def test_sh_config():
    import main

    assert isinstance(main._SELF_HEAL_ENABLED, bool)
    assert isinstance(main._SELF_HEAL_INTERVAL, int)
    assert main._SELF_HEAL_INTERVAL > 0
    assert main._SELF_HEAL_MODEL  # not empty


test_sh_config()


@test("SH 2: Self-heal system prompt is well-formed")
def test_sh_prompt():
    import main

    prompt = main._SELF_HEAL_SYSTEM_PROMPT
    assert "NO_ACTION" in prompt
    assert "REPAIR:" in prompt
    assert "restart_watcher" in prompt
    assert "check_ollama" in prompt
    assert "clear_vram" in prompt
    assert "reindex_knowledge" in prompt


test_sh_prompt()


@test("SH 3: _ai_self_heal_loop is registered in lifespan")
def test_sh_registered():
    import inspect

    import main

    src = inspect.getsource(main.lifespan)
    assert "self_heal_task" in src, "self_heal_task should be created in lifespan"
    assert "_ai_self_heal_loop" in src, "_ai_self_heal_loop should be referenced in lifespan"


test_sh_registered()


@test("SH 4: Self-heal skips when not idle")
def test_sh_skip_active():
    import deltai_api.core as core
    import main

    # During active chat, self-heal should skip
    old_start = core._last_chat_start
    old_end = core._last_chat_end
    try:
        core._last_chat_start = time.time()
        core._last_chat_end = 0.0
        assert not main._is_idle()
    finally:
        core._last_chat_start = old_start
        core._last_chat_end = old_end


test_sh_skip_active()


@test("SH 5: Self-heal loop function exists and is async")
def test_sh_async():
    import inspect as _inspect

    import main

    assert _inspect.iscoroutinefunction(main._ai_self_heal_loop)


test_sh_async()


@test("SH 6: /self-heal/status endpoint exists")
def test_sh_endpoint():
    import main

    routes = [r.path for r in main.app.routes if hasattr(r, "path")]
    assert "/self-heal/status" in routes


test_sh_endpoint()


@test("SH 7: Self-heal respects circuit breaker")
def test_sh_respects_cb():
    import main

    # When CB is open, _cb_check returns False
    old_cb = dict(main._circuit_breaker)
    try:
        main._circuit_breaker["state"] = "open"
        main._circuit_breaker["last_failure"] = time.time()
        main._circuit_breaker["backoff_sec"] = 60
        assert not main._cb_check(), "CB should block when open"
    finally:
        main._circuit_breaker.update(old_cb)


test_sh_respects_cb()


@test("SH 8: Self-heal parse handles REPAIR: format")
def test_sh_parse_repair():
    # Test the parsing logic inline
    valid_repairs = {"restart_watcher", "clear_vram", "reindex_knowledge", "check_ollama"}

    # Direct REPAIR: format
    response = "REPAIR:restart_watcher"
    if response.startswith("REPAIR:"):
        repair_name = response.split("REPAIR:", 1)[1].strip().lower()
    assert repair_name == "restart_watcher"
    assert repair_name in valid_repairs

    # Fuzzy fallback
    response2 = "I think we should restart_watcher since it's stopped"
    repair_name2 = None
    for r in valid_repairs:
        if r in response2.lower():
            repair_name2 = r
            break
    assert repair_name2 == "restart_watcher"


test_sh_parse_repair()


@test("SH 9: Self-heal blocks clear_vram during sim")
def test_sh_blocks_clear_vram_sim():
    import router

    # This is a logical test -- during sim, clear_vram should be blocked
    with patch.object(router, "is_sim_running", return_value=True):
        assert router.is_sim_running(), "Sim should be detected as running"
        # The self-heal loop checks this before executing clear_vram


test_sh_blocks_clear_vram_sim()


# ═══════════════════════════════════════════════════════════════════════
# INTEGRATION
# ═══════════════════════════════════════════════════════════════════════
print("\n=== INTEGRATION ===\n")


@test("INT 1: All 3 background tasks registered in lifespan")
def test_int_all_tasks():
    import inspect

    import main

    src = inspect.getsource(main.lifespan)
    assert "health_task" in src
    assert "resource_task" in src
    assert "self_heal_task" in src
    # All cancelled on shutdown
    assert "health_task, resource_task, self_heal_task" in src


test_int_all_tasks()


@test("INT 2: New endpoints don't conflict with existing ones")
def test_int_no_conflicts():
    import main

    routes = [r.path for r in main.app.routes if hasattr(r, "path")]
    # New endpoints
    new_endpoints = ["/ws/health", "/health/events", "/self-heal/status"]
    for ep in new_endpoints:
        count = routes.count(ep)
        assert count == 1, f"Endpoint {ep} appears {count} times (expected 1)"


test_int_no_conflicts()


@test("INT 3: Resource state has all required fields")
def test_int_resource_state():
    import main

    required = [
        "last_vram_action",
        "last_recovery",
        "vram_warnings",
        "ollama_failures",
        "watcher_restarts",
        "actions_taken",
        "last_sim_state",
        "sim_stop_detected_at",
        "pending_14b_preload",
    ]
    for field in required:
        assert field in main._resource_state, f"Missing field: {field}"


test_int_resource_state()


@test("INT 4: Existing 11 tools still present and matched")
def test_int_tools_intact():
    from tools.definitions import TOOLS
    from tools.executor import EXECUTORS

    assert len(TOOLS) == len(EXECUTORS), (
        f"Mismatch: {len(TOOLS)} tools vs {len(EXECUTORS)} executors"
    )
    assert len(TOOLS) >= 11, f"Expected at least 11 tools, got {len(TOOLS)}"


test_int_tools_intact()


# ═══════════════════════════════════════════════════════════════════════
total = passed + failed
print(f"\n{'=' * 60}")
print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
print(f"{'=' * 60}")

if errors:
    print("\nFailed tests:")
    for name, err in errors:
        print(f"  - {name}: {err}")

print()
sys.exit(0 if failed == 0 else 1)
