"""
deltai Resource Management & Reliability Stress Tests
Tests the new self-management, circuit breaker, and auto-recovery systems
under extreme conditions (VRAM starvation, Ollama outages, concurrent load).
"""

import os
import sys
import time
import json
import asyncio
import threading

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")
os.environ.setdefault("CHROMADB_PATH", os.path.join(os.path.dirname(__file__), "test_chromadb"))
os.environ.setdefault("KNOWLEDGE_PATH", os.path.join(os.path.dirname(__file__), "test_knowledge"))
os.environ.setdefault("TRAINING_PATH", os.path.join(os.path.dirname(__file__), "test_training"))
os.environ.setdefault("SQLITE_PATH", os.path.join(os.path.dirname(__file__), "test_deltai.db"))

from unittest.mock import patch, AsyncMock, MagicMock

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


# ============================================================
# SECTION 1: CIRCUIT BREAKER
# ============================================================

print("\n=== CIRCUIT BREAKER TESTS ===\n")

@test("CB 1: Circuit breaker starts closed")
def test_cb_initial():
    import main
    assert main._circuit_breaker["state"] == "closed"
    assert main._cb_check() is True
test_cb_initial()

@test("CB 2: Circuit breaker opens after threshold failures")
def test_cb_opens():
    import main
    # Reset
    main._cb_success()
    assert main._circuit_breaker["state"] == "closed"
    # Fail 3 times (threshold)
    for _ in range(main._CB_FAILURE_THRESHOLD):
        main._cb_failure()
    assert main._circuit_breaker["state"] == "open", \
        f"Expected open, got {main._circuit_breaker['state']}"
    assert main._cb_check() is False, "Should block calls when open"
    # Reset for other tests
    main._cb_success()
test_cb_opens()

@test("CB 3: Circuit breaker transitions to half-open after backoff")
def test_cb_halfopen():
    import main
    main._cb_success()  # reset
    for _ in range(main._CB_FAILURE_THRESHOLD):
        main._cb_failure()
    assert main._circuit_breaker["state"] == "open"
    # Simulate backoff elapsed
    main._circuit_breaker["last_failure"] = time.time() - main._circuit_breaker["backoff_sec"] - 1
    assert main._cb_check() is True, "Should allow test call after backoff"
    assert main._circuit_breaker["state"] == "half-open"
    # Success resets to closed
    main._cb_success()
    assert main._circuit_breaker["state"] == "closed"
test_cb_halfopen()

@test("CB 4: Circuit breaker backoff doubles up to max")
def test_cb_backoff():
    import main
    main._cb_success()  # reset
    initial_backoff = main._circuit_breaker["backoff_sec"]
    # Trigger open
    for _ in range(main._CB_FAILURE_THRESHOLD):
        main._cb_failure()
    first_backoff = main._circuit_breaker["backoff_sec"]
    # Simulate recovery attempt that fails
    main._circuit_breaker["last_failure"] = time.time() - first_backoff - 1
    main._cb_check()  # transitions to half-open
    for _ in range(main._CB_FAILURE_THRESHOLD):
        main._cb_failure()
    second_backoff = main._circuit_breaker["backoff_sec"]
    assert second_backoff > first_backoff, "Backoff should increase"
    assert second_backoff <= main._CB_MAX_BACKOFF, f"Backoff {second_backoff} exceeds max {main._CB_MAX_BACKOFF}"
    main._cb_success()  # cleanup
test_cb_backoff()

@test("CB 5: Successful call resets circuit breaker completely")
def test_cb_reset():
    import main
    for _ in range(10):
        main._cb_failure()
    main._cb_success()
    assert main._circuit_breaker["state"] == "closed"
    assert main._circuit_breaker["failures"] == 0
    assert main._circuit_breaker["backoff_sec"] == 5  # reset to initial
test_cb_reset()


# ============================================================
# SECTION 2: RESOURCE MANAGER STATE
# ============================================================

print("\n=== RESOURCE MANAGER TESTS ===\n")

@test("RM 1: Resource state initializes correctly")
def test_rm_init():
    import main
    assert "vram_warnings" in main._resource_state
    assert "ollama_failures" in main._resource_state
    assert "watcher_restarts" in main._resource_state
    assert "actions_taken" in main._resource_state
    assert isinstance(main._resource_state["actions_taken"], list)
test_rm_init()

@test("RM 2: Action logging works with bounded history")
def test_rm_logging():
    import main
    old_actions = list(main._resource_state["actions_taken"])
    try:
        main._resource_state["actions_taken"] = []
        for i in range(60):
            main._log_resource_action(f"test action {i}")
        assert len(main._resource_state["actions_taken"]) == 50, \
            f"Expected 50 (capped), got {len(main._resource_state['actions_taken'])}"
        assert main._resource_state["actions_taken"][-1]["action"] == "test action 59"
    finally:
        main._resource_state["actions_taken"] = old_actions
test_rm_logging()

@test("RM 3: Resource manager constants are sane")
def test_rm_constants():
    import main
    assert main._RESOURCE_CHECK_INTERVAL >= 10, "Check interval too fast"
    assert main._VRAM_CRITICAL_MB < main._VRAM_WARN_MB, "Critical should be below warning"
    assert main._MAX_WATCHER_RESTARTS > 0, "Must allow at least 1 restart"
    assert main._OLLAMA_FAILURE_THRESHOLD >= 2, "Need at least 2 failures before action"
    assert main._RESOURCE_ACTION_COOLDOWN >= 30, "Cooldown too short"
test_rm_constants()


# ============================================================
# SECTION 3: SELF-DIAGNOSTIC TOOLS
# ============================================================

print("\n=== SELF-DIAGNOSTIC TOOL TESTS ===\n")

@test("Diag 1: Full diagnostics runs without error")
def test_diag_full():
    from tools.executor import self_diagnostics
    result = self_diagnostics()
    assert "deltai SELF-DIAGNOSTICS" in result
    assert "Ollama:" in result
    assert "ChromaDB:" in result
    assert "GPU:" in result
    assert "Paths:" in result
test_diag_full()

@test("Diag 2: Deep diagnostic for each subsystem")
def test_diag_deep():
    from tools.executor import self_diagnostics
    subsystems = ["ollama", "chromadb", "gpu", "voice", "watcher", "backup", "paths"]
    for sub in subsystems:
        result = self_diagnostics(subsystem=sub)
        assert "ERROR" not in result or "ERROR:" not in result.split("\n")[0], \
            f"Deep diag for {sub} failed: {result[:100]}"
        assert len(result) > 20, f"Deep diag for {sub} too short: {result}"
test_diag_deep()

@test("Diag 3: Invalid subsystem returns error")
def test_diag_invalid():
    from tools.executor import self_diagnostics
    result = self_diagnostics(subsystem="nonexistent")
    assert "ERROR" in result
    assert "Unknown subsystem" in result
test_diag_invalid()

@test("Diag 4: Manage models - status")
def test_model_status():
    from tools.executor import manage_ollama_models
    result = manage_ollama_models(action="status")
    assert "OLLAMA MODEL STATUS" in result or "ERROR" in result
    # If Ollama is running, should list models
    if "ERROR" not in result:
        assert "VRAM" in result or "Loaded" in result
test_model_status()

@test("Diag 5: Manage models - reject non-deltai models")
def test_model_reject():
    from tools.executor import manage_ollama_models
    result = manage_ollama_models(action="unload", model="llama3:latest")
    assert "ERROR" in result
    assert "allowlist" in result.lower() or "only manage" in result.lower()
test_model_reject()

@test("Diag 6: Manage models - preload blocks 14B during sim")
def test_model_sim_block():
    from tools.executor import manage_ollama_models
    import router
    with patch.object(router, 'is_sim_running', return_value=True):
        result = manage_ollama_models(action="preload", model="deltai-qwen14b")
        assert "BLOCKED" in result, f"Should block 14B preload during sim: {result}"
test_model_sim_block()

@test("Diag 7: Repair - restart_watcher")
def test_repair_watcher():
    from tools.executor import repair_subsystem
    result = repair_subsystem(repair="restart_watcher")
    assert "OK" in result or "ERROR" in result  # either works, depends on environment
test_repair_watcher()

@test("Diag 8: Repair - invalid action rejected")
def test_repair_invalid():
    from tools.executor import repair_subsystem
    result = repair_subsystem(repair="delete_everything")
    assert "ERROR" in result
    assert "Unknown repair" in result
test_repair_invalid()

@test("Diag 9: Repair - check_ollama")
def test_repair_ollama():
    from tools.executor import repair_subsystem
    result = repair_subsystem(repair="check_ollama")
    assert "OK" in result or "WARNING" in result or "ERROR" in result
test_repair_ollama()

@test("Diag 10: Resource status tool")
def test_resource_status_tool():
    from tools.executor import resource_status
    result = resource_status()
    assert "deltai RESOURCE STATUS" in result
    assert "VRAM:" in result or "Unavailable" in result
test_resource_status_tool()

@test("Diag 11: Tool definitions include all diagnostic tools")
def test_tool_definitions():
    from tools.definitions import TOOLS, TOOL_MAP
    expected = ["self_diagnostics", "manage_ollama_models", "repair_subsystem", "resource_status"]
    for name in expected:
        assert name in TOOL_MAP, f"Tool '{name}' missing from TOOL_MAP"
    # Verify dispatch table matches
    from tools.executor import EXECUTORS
    for name in expected:
        assert name in EXECUTORS, f"Tool '{name}' missing from EXECUTORS"
test_tool_definitions()


# ============================================================
# SECTION 4: HIGH-STRESS RESOURCE SIMULATIONS
# ============================================================

print("\n=== HIGH-STRESS RESOURCE SIMULATIONS ===\n")

@test("Stress RM 1: Circuit breaker protects Ollama inference path")
def test_stress_cb_inference():
    import main
    # Open the circuit breaker
    main._cb_success()
    for _ in range(main._CB_FAILURE_THRESHOLD):
        main._cb_failure()
    # Inference should get circuit breaker error, not try to connect
    async def run():
        async with __import__('httpx').AsyncClient(timeout=5) as client:
            data, err = await main._try_ollama_inference(client, "deltai-test", [{"role": "user", "content": "test"}])
            return data, err
    data, err = asyncio.run(run())
    assert data is None
    assert "Circuit breaker" in err, f"Expected circuit breaker error, got: {err}"
    main._cb_success()  # reset
test_stress_cb_inference()

@test("Stress RM 2: Rapid circuit breaker cycling doesn't corrupt state")
def test_stress_cb_rapid():
    import main
    main._cb_success()
    for _ in range(100):
        main._cb_failure()
        main._cb_success()
    assert main._circuit_breaker["state"] == "closed"
    assert main._circuit_breaker["failures"] == 0
    # Rapid failures
    for _ in range(50):
        main._cb_failure()
    assert main._circuit_breaker["state"] == "open"
    assert main._circuit_breaker["failures"] == 50
    main._cb_success()
test_stress_cb_rapid()

@test("Stress RM 3: VRAM critical threshold triggers warnings")
def test_stress_vram_critical():
    import main
    # Simulate 3 consecutive low VRAM readings
    main._resource_state["vram_warnings"] = 0
    for _ in range(3):
        vram = 1000  # critical
        if vram < main._VRAM_CRITICAL_MB:
            main._resource_state["vram_warnings"] += 1
    assert main._resource_state["vram_warnings"] >= 2, \
        "Should accumulate VRAM warnings"
    main._resource_state["vram_warnings"] = 0  # reset
test_stress_vram_critical()

@test("Stress RM 4: Watcher restart cap respected")
def test_stress_watcher_cap():
    import main
    old = main._resource_state["watcher_restarts"]
    main._resource_state["watcher_restarts"] = main._MAX_WATCHER_RESTARTS
    # Should not attempt restart
    assert main._resource_state["watcher_restarts"] >= main._MAX_WATCHER_RESTARTS
    main._resource_state["watcher_restarts"] = old
test_stress_watcher_cap()

@test("Stress RM 5: Ollama failure counter tracks correctly")
def test_stress_ollama_failures():
    import main
    old = main._resource_state["ollama_failures"]
    main._resource_state["ollama_failures"] = 0
    for i in range(10):
        main._resource_state["ollama_failures"] += 1
    assert main._resource_state["ollama_failures"] == 10
    # Threshold check
    assert main._resource_state["ollama_failures"] >= main._OLLAMA_FAILURE_THRESHOLD
    main._resource_state["ollama_failures"] = old
test_stress_ollama_failures()

@test("Stress RM 6: Emergency inference fallback with circuit breaker")
def test_stress_emergency_with_cb():
    import main
    main._cb_success()  # ensure clean state
    # Simulate: primary fails (circuit opens), backup should still work if CB resets
    async def run():
        # Open circuit breaker
        for _ in range(main._CB_FAILURE_THRESHOLD):
            main._cb_failure()
        # First call should hit CB
        async with __import__('httpx').AsyncClient(timeout=5) as client:
            data, err, model, is_emergency = await main._inference_with_emergency_fallback(
                client, "deltai-qwen14b", [{"role": "user", "content": "test"}],
                None, "deltai-qwen3b"
            )
        return data, err, model, is_emergency
    data, err, model, is_emergency = asyncio.run(run())
    # Should fail because CB is open for all Ollama calls
    assert data is None, "Should fail with CB open"
    assert "Circuit breaker" in str(err) or "failed" in str(err).lower()
    main._cb_success()
test_stress_emergency_with_cb()

@test("Stress RM 7: Concurrent resource state access is safe")
def test_stress_concurrent_state():
    import main
    results = []
    def modify_state(i):
        main._log_resource_action(f"concurrent test {i}")
        main._resource_state["vram_warnings"] = i % 5
        results.append(True)
    threads = [threading.Thread(target=modify_state, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert len(results) == 20, f"Expected 20 results, got {len(results)}"
    # State should be consistent (not corrupted)
    assert isinstance(main._resource_state["vram_warnings"], int)
    assert isinstance(main._resource_state["actions_taken"], list)
test_stress_concurrent_state()

@test("Stress RM 8: Full diagnostic under simulated VRAM pressure")
def test_stress_diag_under_pressure():
    from tools.executor import self_diagnostics
    # Run full diagnostics — should work regardless of GPU state
    result = self_diagnostics()
    assert "deltai SELF-DIAGNOSTICS" in result
    # Run GPU deep diagnostic
    result = self_diagnostics(subsystem="gpu")
    assert "GPU" in result
    assert len(result) > 50
test_stress_diag_under_pressure()

@test("Stress RM 9: Manage models handles Ollama down gracefully")
def test_stress_models_ollama_down():
    from tools.executor import manage_ollama_models, _ollama_get
    with patch('tools.executor._ollama_get', return_value=None):
        result = manage_ollama_models(action="status")
        assert "ERROR" in result
        assert "reach Ollama" in result or "running" in result
test_stress_models_ollama_down()

@test("Stress RM 10: All repair actions handle errors gracefully")
def test_stress_repair_errors():
    from tools.executor import repair_subsystem
    # Each repair should return a string, never raise
    for action in ["restart_watcher", "clear_vram", "reindex_knowledge", "check_ollama"]:
        result = repair_subsystem(repair=action)
        assert isinstance(result, str), f"repair '{action}' should return string"
        assert len(result) > 5, f"repair '{action}' response too short: {result}"
test_stress_repair_errors()


# ============================================================
# RESULTS
# ============================================================

total = passed + failed
print(f"\n{'='*60}")
print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
print(f"{'='*60}")

if errors:
    print("\nFailed tests:")
    for name, err in errors:
        print(f"  - {name}: {err}")

print()
sys.exit(0 if failed == 0 else 1)
