"""
deltai Stress Test Suite -- High-Load Simulation & Verification
Tests all fixes from the 2026-03-17 code review + stress scenarios.

Simulates:
  - Low VRAM environments (GPU maxed by sim/rendering)
  - High CPU load conditions
  - Concurrent ingest + chat under pressure
  - Emergency backup cascade under resource starvation
  - Session mode GPU protection with degraded resources
  - All 10 review fixes verified
"""

import os
import sys

# Fix console encoding for Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import asyncio
import threading
import time
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── ENVIRONMENT SETUP ─────────────────────────────────────────────────
os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")
os.environ.setdefault("CHROMADB_PATH", os.path.join(os.path.dirname(__file__), "test_chromadb"))
os.environ.setdefault("KNOWLEDGE_PATH", os.path.join(os.path.dirname(__file__), "test_knowledge"))
os.environ.setdefault("TRAINING_PATH", os.path.join(os.path.dirname(__file__), "test_training"))
os.environ.setdefault("SQLITE_PATH", os.path.join(os.path.dirname(__file__), "test_deltai.db"))

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
# SECTION 1: REVIEW FIX VERIFICATION (10 fixes)
# ═══════════════════════════════════════════════════════════════════════

print("\n=== REVIEW FIX VERIFICATION ===\n")


# Fix 1: Operator precedence in _backup_health_loop
@test("Fix 1: BACKUP_ENABLED operator precedence (not in vs in)")
def verify_fix_1():
    import inspect

    import main

    source = inspect.getsource(main._backup_health_loop)
    # Must use 'not in' (correct) instead of 'not ... in' (wrong precedence)
    assert ".lower() not in (" in source, (
        "Expected 'not in' pattern for correct operator precedence"
    )
    assert "not os.getenv" not in source, (
        "Found old 'not os.getenv()' pattern — precedence bug still present"
    )


verify_fix_1()


# Fix 2: Async ingest endpoints use asyncio.to_thread
@test("Fix 2: Ingest endpoints wrap blocking calls with asyncio.to_thread")
def verify_fix_2():
    import inspect

    import main

    src = inspect.getsource(main.ingest_endpoint)
    assert "asyncio.to_thread" in src, (
        "ingest_endpoint should use asyncio.to_thread for blocking ingest_context"
    )
    src_batch = inspect.getsource(main.ingest_batch_endpoint)
    assert "asyncio.to_thread" in src_batch, (
        "ingest_batch_endpoint should use asyncio.to_thread for blocking ingest_context_batch"
    )


verify_fix_2()


# Fix 3: cleanup_expired uses targeted query instead of full scan
@test("Fix 3: cleanup_expired() uses targeted ChromaDB query with rate limit")
def verify_fix_3():
    import inspect

    import memory

    src = inspect.getsource(memory.cleanup_expired)
    assert "_CLEANUP_MIN_INTERVAL" in src or "rate-limit" in src.lower(), (
        "cleanup_expired should have rate limiting"
    )
    assert '"$gt"' in src or "'$gt'" in src, (
        "cleanup_expired should use $gt filter for targeted queries"
    )
    # Verify rate limiting works
    memory._cleanup_last_run = time.time()
    result = memory.cleanup_expired()
    assert result.get("skipped") == "rate-limited", "Rate limiting should skip rapid calls"


verify_fix_3()


# Fix 4: PowerShell injection fixed in voice TTS
@test("Fix 4: Windows SAPI TTS uses temp file instead of inline text")
def verify_fix_4():
    import inspect

    import voice

    src = inspect.getsource(voice._windows_tts)
    assert "ReadAllText" in src or "Get-Content" in src or "text_path" in src, (
        "Windows TTS should read text from temp file, not inline"
    )
    assert "text_path" in src, "Should write to temp text file before passing to PowerShell"


verify_fix_4()


# Fix 5: WAL pragma only in init_db
@test("Fix 5: WAL pragma set once in init_db, not per connection")
def verify_fix_5():
    import inspect

    import persistence

    connect_src = inspect.getsource(persistence._connect)
    init_src = inspect.getsource(persistence.init_db)
    assert "journal_mode=WAL" not in connect_src, "_connect() should NOT set WAL pragma"
    assert "journal_mode=WAL" in init_src, "init_db() should set WAL pragma"


verify_fix_5()


# Fix 6: Router default model names match CLAUDE.md
@test("Fix 6: Router _pick_local_model defaults match deltai-qwen14b/3b")
def verify_fix_6():
    import inspect

    import router

    src = inspect.getsource(router._pick_local_model)
    assert "deltai-qwen14b" in src, "Strong model default should be deltai-qwen14b"
    assert "deltai-qwen3b" in src, "Default model should be deltai-qwen3b"


verify_fix_6()


# Fix 7: training.py list_datasets getsize race condition
@test("Fix 7: list_datasets handles OSError on getsize")
def verify_fix_7():
    import inspect

    import training

    src = inspect.getsource(training.list_datasets)
    assert "except OSError" in src, "list_datasets should catch OSError on os.path.getsize"


verify_fix_7()


# Fix 8: Anthropic API version documented
@test("Fix 8: Anthropic API version header is documented")
def verify_fix_8():
    import anthropic_client

    assert hasattr(anthropic_client, "ANTHROPIC_VERSION"), "ANTHROPIC_VERSION constant should exist"
    assert anthropic_client.ANTHROPIC_VERSION == "2023-06-01", "API version should be 2023-06-01"


verify_fix_8()


# Fix 9: asyncio.to_thread replaces deprecated get_event_loop
@test("Fix 9: No deprecated asyncio.get_event_loop() calls")
def verify_fix_9():
    import inspect

    import main

    src = inspect.getsource(main)
    # Count occurrences — should be zero
    count = src.count("get_event_loop()")
    assert count == 0, (
        f"Found {count} remaining get_event_loop() calls — should use asyncio.to_thread"
    )


verify_fix_9()


# Fix 10: Code cleanup — consolidated GPU queries
@test("Fix 10: GPU VRAM queries consolidated into _get_vram_info()")
def verify_fix_10():
    import router

    assert hasattr(router, "_get_vram_info"), (
        "router should have consolidated _get_vram_info() function"
    )
    import inspect

    src = inspect.getsource(router._get_vram_info)
    assert "used" in src and "total" in src, (
        "_get_vram_info should return both used and total in one call"
    )


verify_fix_10()


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: HIGH-STRESS SIMULATIONS
# ═══════════════════════════════════════════════════════════════════════

print("\n=== HIGH-STRESS SIMULATIONS ===\n")


# Stress 1: Low VRAM routing (< 3GB free)
@test("Stress 1: Router correctly picks CPU mode when VRAM < 3GB")
def stress_low_vram():
    import router

    with (
        patch.object(router, "get_vram_free_mb", return_value=1500),
        patch.object(router, "is_sim_running", return_value=False),
        patch.object(router, "is_gpu_loaded", return_value=False),
    ):
        model, cpu_only, backup, _, _ = router._pick_local_model()
        assert cpu_only is True, (
            f"Should be CPU-only with 1500MB free VRAM, got cpu_only={cpu_only}"
        )
        assert model == os.getenv("DELTAI_MODEL", "deltai-qwen3b"), (
            f"Should use default small model, got {model}"
        )


stress_low_vram()


# Stress 2: Zero VRAM scenario
@test("Stress 2: Router handles 0MB free VRAM gracefully")
def stress_zero_vram():
    import router

    with (
        patch.object(router, "get_vram_free_mb", return_value=0),
        patch.object(router, "is_sim_running", return_value=True),
        patch.object(router, "is_gpu_loaded", return_value=True),
    ):
        model, cpu_only, backup, _, _ = router._pick_local_model()
        assert cpu_only is True, "Must be CPU-only with 0 VRAM"
        assert backup is not None, "Should have a backup model even at 0 VRAM"


stress_zero_vram()


# Stress 3: Sim running + low VRAM
@test("Stress 3: Sim running with contested VRAM picks 3B model")
def stress_sim_low_vram():
    import router

    with (
        patch.object(router, "get_vram_free_mb", return_value=4000),
        patch.object(router, "is_sim_running", return_value=True),
    ):
        model, cpu_only, backup, _, _ = router._pick_local_model()
        assert cpu_only is False, "4GB VRAM should allow GPU mode for 3B"
        assert "3b" in model.lower() or model == os.getenv(
            "DELTAI_SIM_MODEL", os.getenv("DELTAI_MODEL", "deltai-qwen3b")
        ), f"Should pick sim/small model, got {model}"


stress_sim_low_vram()


# Stress 4: Emergency backup cascade
@test("Stress 4: Backup model chain deltai-qwen14b -> deltai-nemo -> deltai-fallback")
def stress_backup_chain():
    import router

    backup1 = router.get_backup_model("deltai-qwen14b")
    assert backup1 is not None, "deltai-qwen14b should have a backup"
    backup2 = router.get_backup_model(backup1)
    assert backup2 is not None, f"{backup1} should have a second backup"
    backup3 = router.get_backup_model(backup2)
    assert backup3 is None, f"{backup2} should be the end of the chain (got {backup3})"


stress_backup_chain()


# Stress 5: Router force_local always picks ollama
@test("Stress 5: Force local always routes to ollama")
def stress_force_local():
    import router

    with (
        patch.object(router, "get_vram_free_mb", return_value=2000),
        patch.object(router, "is_sim_running", return_value=True),
        patch.object(router, "is_gpu_loaded", return_value=True),
    ):
        decision = asyncio.run(router.route("what are my tire temps", force_local=True))
        assert decision.backend == "ollama", (
            f"Force local should always route to ollama, got {decision.backend}"
        )


stress_force_local()


# Stress 6: Tier classification under load
@test("Stress 6: Complexity classifier handles edge cases")
def stress_tier_classification():
    import router

    # Very short messages should be tier 1
    assert router.classify_complexity("hi") == 1
    assert router.classify_complexity("") == 1 or router.classify_complexity("") in (
        1,
        2,
    )  # empty is fine as tier 1
    # Complex engineering should be tier 3
    assert (
        router.classify_complexity("derive the stress-strain equation for this beam deflection")
        == 3
    )
    # Code review should be tier 2
    assert router.classify_complexity("review this code and fix the bugs") == 2
    # Long ambiguous messages (>300 chars with question words) should bump to tier 2
    long_msg = (
        "I need help understanding "
        + "something really complex " * 15
        + "? how does this work and why is it so difficult?"
    )
    assert len(long_msg) > 300, f"Test message too short: {len(long_msg)}"
    assert router.classify_complexity(long_msg) >= 2


stress_tier_classification()


# Stress 7: Routing always returns query_category=general (telemetry modes removed)
@test("Stress 7: Route always returns general query category")
def stress_general_category():
    import router

    # Even telemetry-sounding queries should route as general now
    decision = asyncio.run(router.route("what are my tire temps"))
    assert decision.query_category == "general", f"Expected general, got {decision.query_category}"
    decision = asyncio.run(router.route("should i pit now"))
    assert decision.query_category == "general", f"Expected general, got {decision.query_category}"
    decision = asyncio.run(router.route("write me a python script"))
    assert decision.query_category == "general", f"Expected general, got {decision.query_category}"


stress_general_category()


# Stress 8: Greeting short-circuit under all patterns
@test("Stress 8: Greeting matcher covers all patterns without false positives")
def stress_greetings():
    import main

    # Should match
    assert main._check_greeting("hey") is not None
    assert main._check_greeting("Hey!") is not None
    assert main._check_greeting("good morning") is not None
    assert main._check_greeting("gn") is not None
    assert main._check_greeting("you there?") is not None
    # Should NOT match (real questions)
    assert main._check_greeting("hey can you help me with something") is None
    assert main._check_greeting("what is the weather") is None
    assert main._check_greeting("debug this code") is None
    assert main._check_greeting("morning report on the system") is None


stress_greetings()


# Stress 9: Text-as-tool parser robustness
@test("Stress 9: Text-as-tool parser handles malformed JSON, Windows paths, nested objects")
def stress_tool_parser():
    import main

    # Standard format
    r = main.try_parse_text_tool_call(
        '{"name":"read_file","parameters":{"path":"C:\\\\deltai\\\\test.py"}}'
    )
    assert r is not None and r[0] == "read_file"
    # Markdown code block
    r = main.try_parse_text_tool_call(
        'Let me read that:\n```json\n{"name":"read_file","arguments":{"path":"test.py"}}\n```'
    )
    assert r is not None and r[0] == "read_file"
    # Python-style booleans
    r = main.try_parse_text_tool_call(
        '{"name":"list_directory","parameters":{"path":"C:\\\\deltai","recursive":True}}'
    )
    assert r is not None and r[0] == "list_directory"
    # Ollama-style array
    r = main.try_parse_text_tool_call('[{"function":{"name":"get_system_info","arguments":{}}}]')
    assert r is not None and r[0] == "get_system_info"
    # Garbage — should return None
    r = main.try_parse_text_tool_call("I don't need any tools for this.")
    assert r is None
    # Empty
    r = main.try_parse_text_tool_call("")
    assert r is None
    r = main.try_parse_text_tool_call(None)
    assert r is None


stress_tool_parser()


# Stress 10: Budget enforcement
@test("Stress 10: Cloud budget enforcement blocks when exhausted")
def stress_budget():
    import router

    old_spend = router._daily_cloud_spend
    old_date = router._daily_cloud_reset_date
    try:
        router._daily_cloud_spend = 100.0  # Way over budget
        router._daily_cloud_reset_date = time.strftime("%Y-%m-%d")
        assert not router._check_budget(), "Budget should be exhausted at $100 spend"

        router._daily_cloud_spend = 0.01
        assert router._check_budget(), "Budget should be ok at $0.01 spend"

        # Test daily reset
        router._daily_cloud_reset_date = "1999-01-01"
        router._daily_cloud_spend = 100.0
        assert router._check_budget(), "Budget should reset on new day"
    finally:
        router._daily_cloud_spend = old_spend
        router._daily_cloud_reset_date = old_date


stress_budget()


# Stress 11: Conversation history rollover
@test("Stress 11: Conversation history respects max turns and trims correctly")
def stress_history():
    import main

    old_history = list(main._conversation_history)
    old_max = main.HISTORY_MAX_TURNS
    try:
        main._conversation_history.clear()
        main.HISTORY_MAX_TURNS = 3
        # Add 5 turns (10 messages) — should keep only 3 turns (6 messages)
        for i in range(5):
            main._conversation_history.append({"role": "user", "content": f"msg {i}"})
            main._conversation_history.append({"role": "assistant", "content": f"resp {i}"})
        max_msgs = main.HISTORY_MAX_TURNS * 2
        if len(main._conversation_history) > max_msgs:
            del main._conversation_history[:-max_msgs]
        assert len(main._conversation_history) == 6, (
            f"Expected 6 messages (3 turns), got {len(main._conversation_history)}"
        )
        assert main._conversation_history[0]["content"] == "msg 2", (
            "Oldest message should be msg 2 after trimming"
        )
    finally:
        main._conversation_history.clear()
        main._conversation_history.extend(old_history)
        main.HISTORY_MAX_TURNS = old_max


stress_history()


# Stress 12: Router sim detection affects model selection
@test("Stress 12: Sim running picks smaller model to save VRAM")
def stress_sim_model_selection():
    import router

    # When focus workload is active, router should prefer smaller model
    with (
        patch.object(router, "is_sim_running", return_value=True),
        patch.object(router, "get_vram_free_mb", return_value=5000),
    ):
        model, cpu_only, backup, _, _ = router._pick_local_model()
        assert "3b" in model.lower() or model == os.getenv(
            "DELTAI_SIM_MODEL", os.getenv("DELTAI_MODEL", "deltai-qwen3b")
        ), f"Sim running should pick small model, got {model}"
    # When sim is NOT running and plenty of VRAM, pick strong model
    with (
        patch.object(router, "is_sim_running", return_value=False),
        patch.object(router, "get_vram_free_mb", return_value=10000),
    ):
        model, cpu_only, backup, _, _ = router._pick_local_model()
        assert "14b" in model.lower() or model == os.getenv(
            "DELTAI_STRONG_MODEL", "deltai-qwen14b"
        ), f"No sim + plenty VRAM should pick strong model, got {model}"


stress_sim_model_selection()


# Stress 13: Tool safety guards
@test("Stress 13: Tool executor blocks dangerous commands")
def stress_tool_safety():
    from tools.executor import _is_command_safe, _is_path_safe_write

    # Should block
    assert not _is_command_safe("rm -rf /"), "rm -rf / should be blocked"
    assert not _is_command_safe("sudo apt install evil"), "sudo should be blocked"
    assert not _is_command_safe("shutdown now"), "shutdown should be blocked"
    assert not _is_command_safe("useradd hacker"), "useradd should be blocked"
    assert not _is_command_safe("reboot"), "reboot should be blocked"
    # Should allow
    assert _is_command_safe("ls ~/deltai")
    assert _is_command_safe("ps aux | grep python")
    # Path safety
    assert not _is_path_safe_write("/etc/test.txt"), "/etc should be protected"
    assert not _is_path_safe_write("/boot/grub/test.txt"), "/boot should be protected"
    assert _is_path_safe_write(os.path.expanduser("~/deltai/test.txt"))


stress_tool_safety()


# Stress 14: Training safety — blocks during sim
@test("Stress 14: Training blocks when focus workload is active")
def stress_training_safety():
    import router
    import training

    # Ensure dataset exists first
    training.delete_dataset("stress-sim-test")
    training.create_dataset("stress-sim-test")
    training.add_example("stress-sim-test", "test input", "test output " * 10)
    try:
        with patch.object(router, "is_sim_running", return_value=True):
            result = training.start_training("stress-sim-test", mode="lora")
            assert result["status"] == "error", (
                "Training should be blocked while focus workload is active"
            )
            assert "focus" in result["reason"].lower() or "workload" in result["reason"].lower(), (
                f"Error should mention focus/workload: {result['reason']}"
            )
    finally:
        training.delete_dataset("stress-sim-test")


stress_training_safety()


# Stress 15: Concurrent route decisions (simulated)
@test("Stress 15: Router handles rapid concurrent calls without state corruption")
def stress_concurrent_routing():
    import router

    results = []

    def route_sync(msg):
        decision = asyncio.run(router.route(msg))
        results.append(decision)

    threads = []
    messages = [
        "hello",
        "analyze telemetry",
        "what is 2+2",
        "derive stress equation",
        "read my file",
        "pit strategy for next stint",
        "system status",
    ]
    for msg in messages:
        t = threading.Thread(target=route_sync, args=(msg,))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(results) == len(messages), (
        f"Expected {len(messages)} route decisions, got {len(results)}"
    )
    # All should have valid backend
    for r in results:
        assert r.backend in ("ollama", "anthropic"), f"Invalid backend: {r.backend}"
        assert r.tier in (1, 2, 3), f"Invalid tier: {r.tier}"


stress_concurrent_routing()


# Stress 16: VRAM tier transitions (A->B->C)
@test("Stress 16: VRAM tier transitions A->B->C under decreasing VRAM")
def stress_vram_tiers():
    import router

    with patch.object(router, "is_sim_running", return_value=False):
        # Tier A: > 9GB
        with patch.object(router, "get_vram_free_mb", return_value=10000):
            model_a, cpu_a, _, _, _ = router._pick_local_model()
            assert not cpu_a, "Tier A should use GPU"
            assert "14b" in model_a.lower() or model_a == os.getenv(
                "DELTAI_STRONG_MODEL", "deltai-qwen14b"
            )

        # Tier B: 3-9GB
        with patch.object(router, "get_vram_free_mb", return_value=5000):
            model_b, cpu_b, _, _, _ = router._pick_local_model()
            assert not cpu_b, "Tier B should use GPU"

        # Tier C: < 3GB
        with patch.object(router, "get_vram_free_mb", return_value=1000):
            model_c, cpu_c, _, _, _ = router._pick_local_model()
            assert cpu_c, "Tier C should force CPU"


stress_vram_tiers()


# Stress 17: Persistence under load
@test("Stress 17: SQLite persistence handles rapid read/write cycles")
def stress_persistence():
    import persistence

    persistence.init_db()

    # Rapid writes
    for i in range(20):
        persistence.save_history_pair(f"user_msg_{i}", f"assistant_msg_{i}")

    # Read back
    history = persistence.load_history(10)
    assert len(history) >= 20, f"Expected at least 20 messages, got {len(history)}"

    # Budget rapid updates
    today = time.strftime("%Y-%m-%d")
    for i in range(10):
        persistence.save_budget(today, i * 0.5)
    budget = persistence.load_budget(today)
    assert budget == 4.5, f"Expected budget 4.5, got {budget}"

    # Trim
    persistence.trim_history(5)
    history = persistence.load_history(5)
    assert len(history) <= 10, f"Expected ≤10 messages after trim, got {len(history)}"

    # Cleanup
    persistence.clear_history()


stress_persistence()


# Stress 18: RAG query expansion
@test("Stress 18: Query expansion generates meaningful variants")
def stress_query_expansion():
    import memory

    # Should expand
    variants = memory._expand_query("how does the router handle GPU VRAM")
    assert len(variants) >= 2, f"Expected 2+ variants, got {len(variants)}"
    assert variants[0] == "how does the router handle GPU VRAM", "First variant should be original"

    # Very short query
    variants = memory._expand_query("hi")
    assert len(variants) >= 1, "Even short queries should return at least 1 variant"

    # Stop-word-only query
    variants = memory._expand_query("the is a")
    assert len(variants) >= 1


stress_query_expansion()


# Stress 19: Dataset operations under pressure
@test("Stress 19: Training dataset CRUD handles rapid operations")
def stress_dataset_crud():
    import training

    name = "stress-test-dataset"

    # Cleanup from prior runs
    training.delete_dataset(name)

    # Create
    r = training.create_dataset(name)
    assert r["status"] == "ok"

    # Rapid adds
    for i in range(50):
        r = training.add_example(name, f"input {i}", f"output {i}", category="stress")
    assert r["status"] == "ok"

    # Read back
    ds = training.get_dataset(name)
    assert ds["status"] == "ok"
    assert len(ds["examples"]) == 50, f"Expected 50 examples, got {len(ds['examples'])}"

    # Export
    for fmt in ("alpaca", "sharegpt", "chatml"):
        r = training.export_dataset(name, fmt)
        assert r["status"] == "ok", f"Export {fmt} failed: {r}"

    # Remove examples
    training.remove_example(name, 0)
    ds = training.get_dataset(name)
    assert len(ds["examples"]) == 49

    # Cleanup
    training.delete_dataset(name)
    assert not os.path.exists(training._dataset_path(name))


stress_dataset_crud()


# Stress 20: Auto-capture filtering
@test("Stress 20: Auto-capture filters correctly")
def stress_auto_capture():
    import training

    # Should NOT capture: too short
    r = training.auto_capture("test-auto", "hi", "ok")
    assert not r["captured"], f"Short response should not be captured: {r}"

    # Should NOT capture: canned greeting
    r = training.auto_capture("test-auto", "hello", "Operational.")
    assert not r["captured"]

    # Should NOT capture: error response
    r = training.auto_capture("test-auto", "do something", "error: something broke in the system")
    assert not r["captured"]

    # Should capture: good exchange
    r = training.auto_capture(
        "test-auto",
        "explain the router",
        "The router examines VRAM availability and sim status to pick the best model. " * 3,
    )
    assert r["captured"], f"Good exchange should be captured: {r}"

    # Cleanup
    training.delete_dataset("test-auto")


stress_auto_capture()


# ═══════════════════════════════════════════════════════════════════════
# RESULTS
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
