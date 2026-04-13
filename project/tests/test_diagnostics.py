"""
deltai Self-Diagnostic Tools — Test Suite
Tests the 3 new tools: self_diagnostics, manage_ollama_models, repair_subsystem
"""

import os
import sys

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


print("\n=== DIAGNOSTIC TOOL REGISTRATION ===\n")


@test("Tools registered in TOOLS list")
def test_tools_registered():
    from tools.definitions import TOOL_MAP, TOOLS

    tool_names = [t["function"]["name"] for t in TOOLS]
    assert "self_diagnostics" in tool_names, "self_diagnostics missing from TOOLS"
    assert "manage_ollama_models" in tool_names, "manage_ollama_models missing from TOOLS"
    assert "repair_subsystem" in tool_names, "repair_subsystem missing from TOOLS"
    assert "self_diagnostics" in TOOL_MAP, "self_diagnostics missing from TOOL_MAP"
    assert "manage_ollama_models" in TOOL_MAP, "manage_ollama_models missing from TOOL_MAP"
    assert "repair_subsystem" in TOOL_MAP, "repair_subsystem missing from TOOL_MAP"


test_tools_registered()


@test("Executors registered in EXECUTORS dict")
def test_executors_registered():
    from tools.executor import EXECUTORS

    assert "self_diagnostics" in EXECUTORS, "self_diagnostics missing from EXECUTORS"
    assert "manage_ollama_models" in EXECUTORS, "manage_ollama_models missing from EXECUTORS"
    assert "repair_subsystem" in EXECUTORS, "repair_subsystem missing from EXECUTORS"


test_executors_registered()


@test("execute_tool dispatches to new tools")
def test_execute_dispatch():
    from tools.executor import execute_tool

    # self_diagnostics should return a string (not ERROR: Unknown tool)
    result = execute_tool("self_diagnostics", {})
    assert "Unknown tool" not in result, f"self_diagnostics not dispatched: {result[:100]}"
    result = execute_tool("manage_ollama_models", {"action": "status"})
    assert "Unknown tool" not in result, f"manage_ollama_models not dispatched: {result[:100]}"


test_execute_dispatch()


@test("Tool count is 10 core (7 original + 3 diagnostic)")
def test_tool_count():
    from tools.definitions import TOOLS

    # Count non-telemetry tools (telemetry tools are conditional)
    core_names = {
        "read_file",
        "write_file",
        "list_directory",
        "run_shell",
        "get_system_info",
        "search_knowledge",
        "memory_stats",
        "self_diagnostics",
        "manage_ollama_models",
        "repair_subsystem",
    }
    found = {t["function"]["name"] for t in TOOLS if t["function"]["name"] in core_names}
    assert found == core_names, f"Expected 10 core tools, found: {found}"


test_tool_count()


print("\n=== SELF_DIAGNOSTICS TESTS ===\n")


@test("self_diagnostics() returns structured report with all subsystems")
def test_diag_full():
    from tools.executor import self_diagnostics

    result = self_diagnostics()
    assert "deltai SELF-DIAGNOSTICS" in result, "Missing header"
    # Should mention all subsystems
    for subsystem in ["Ollama", "ChromaDB", "GPU", "Voice", "Watcher", "Paths", "Backup"]:
        assert subsystem in result or subsystem.lower() in result.lower(), (
            f"Missing subsystem: {subsystem}"
        )


test_diag_full()


@test("self_diagnostics(subsystem='ollama') returns deep Ollama report")
def test_diag_ollama():
    from tools.executor import self_diagnostics

    result = self_diagnostics(subsystem="ollama")
    assert "OLLAMA" in result.upper(), "Missing Ollama header"
    # Should mention models or DOWN status
    assert "model" in result.lower() or "DOWN" in result, "Should mention models or DOWN status"


test_diag_ollama()


@test("self_diagnostics(subsystem='gpu') returns GPU details")
def test_diag_gpu():
    from tools.executor import self_diagnostics

    result = self_diagnostics(subsystem="gpu")
    assert "GPU" in result.upper(), "Missing GPU header"
    # Should have VRAM or unavailable
    assert "VRAM" in result or "unavailable" in result.lower(), "Should mention VRAM or unavailable"


test_diag_gpu()


@test("self_diagnostics(subsystem='chromadb') returns ChromaDB details")
def test_diag_chromadb():
    from tools.executor import self_diagnostics

    result = self_diagnostics(subsystem="chromadb")
    assert "CHROMADB" in result.upper(), "Missing ChromaDB header"


test_diag_chromadb()


@test("self_diagnostics(subsystem='paths') checks critical paths")
def test_diag_paths():
    from tools.executor import self_diagnostics

    result = self_diagnostics(subsystem="paths")
    assert "PATH" in result.upper(), "Missing paths header"
    assert "deltai" in result.lower(), "Should reference deltai paths"


test_diag_paths()


@test("self_diagnostics(subsystem='watcher') checks watcher")
def test_diag_watcher():
    from tools.executor import self_diagnostics

    result = self_diagnostics(subsystem="watcher")
    assert "WATCHER" in result.upper(), "Missing watcher header"


test_diag_watcher()


@test("self_diagnostics(subsystem='backup') checks backup chain")
def test_diag_backup():
    from tools.executor import self_diagnostics

    result = self_diagnostics(subsystem="backup")
    assert "BACKUP" in result.upper(), "Missing backup header"


test_diag_backup()


@test("self_diagnostics(subsystem='invalid') returns error")
def test_diag_invalid():
    from tools.executor import self_diagnostics

    result = self_diagnostics(subsystem="nonexistent")
    assert "ERROR" in result or "Unknown" in result, (
        f"Expected error for invalid subsystem, got: {result[:100]}"
    )


test_diag_invalid()


print("\n=== MANAGE_OLLAMA_MODELS TESTS ===\n")


@test("manage_ollama_models('status') returns model list")
def test_models_status():
    from tools.executor import manage_ollama_models

    result = manage_ollama_models(action="status")
    # Should either show models or report Ollama unreachable
    assert "MODEL STATUS" in result or "ERROR" in result, (
        f"Expected status report or error: {result[:100]}"
    )
    if "ERROR" not in result:
        assert "VRAM" in result or "Loaded" in result or "Available" in result, (
            "Status should contain model/VRAM info"
        )


test_models_status()


@test("manage_ollama_models('unload') requires model parameter")
def test_models_unload_no_model():
    from tools.executor import manage_ollama_models

    result = manage_ollama_models(action="unload")
    assert "ERROR" in result and "model" in result.lower(), (
        f"Should require model parameter: {result}"
    )


test_models_unload_no_model()


@test("manage_ollama_models rejects non-deltai models")
def test_models_allowlist():
    from tools.executor import manage_ollama_models

    result = manage_ollama_models(action="unload", model="llama3:latest")
    assert "ERROR" in result and "allowlist" in result.lower(), (
        f"Should reject non-deltai model: {result}"
    )
    result = manage_ollama_models(action="preload", model="gpt-4")
    assert "ERROR" in result and "allowlist" in result.lower(), (
        f"Should reject non-deltai model: {result}"
    )


test_models_allowlist()


@test("manage_ollama_models accepts deltai model names")
def test_models_deltai_accepted():
    from tools.executor import manage_ollama_models

    # Unload should not fail with allowlist error for deltai models
    for model in ["deltai-qwen14b", "deltai-qwen3b", "deltai-nemo", "deltai-fallback", "deltai"]:
        result = manage_ollama_models(action="unload", model=model)
        assert "allowlist" not in result.lower(), (
            f"deltai model {model} should be accepted: {result}"
        )


test_models_deltai_accepted()


@test("manage_ollama_models rejects invalid action")
def test_models_invalid_action():
    from tools.executor import manage_ollama_models

    result = manage_ollama_models(action="delete")
    assert "ERROR" in result and "Unknown action" in result, (
        f"Should reject invalid action: {result}"
    )


test_models_invalid_action()


@test("manage_ollama_models blocks 14B preload during sim")
def test_models_sim_guard():
    from unittest.mock import patch

    import router
    from tools.executor import manage_ollama_models

    with patch.object(router, "is_sim_running", return_value=True):
        result = manage_ollama_models(action="preload", model="deltai-qwen14b")
        assert "BLOCKED" in result or "sim" in result.lower(), (
            f"Should block 14B preload during sim: {result}"
        )


test_models_sim_guard()


print("\n=== REPAIR_SUBSYSTEM TESTS ===\n")


@test("repair_subsystem('restart_watcher') restarts watcher")
def test_repair_watcher():
    from tools.executor import repair_subsystem

    result = repair_subsystem(repair="restart_watcher")
    assert "OK" in result or "ERROR" in result, f"Should return OK or ERROR: {result}"
    if "OK" in result:
        assert "restart" in result.lower(), "Should confirm restart"


test_repair_watcher()


@test("repair_subsystem('check_ollama') checks Ollama")
def test_repair_ollama():
    from tools.executor import repair_subsystem

    result = repair_subsystem(repair="check_ollama")
    # Should either confirm running or try to start
    assert "OK" in result or "WARNING" in result or "ERROR" in result, (
        f"Unexpected result: {result}"
    )


test_repair_ollama()


@test("repair_subsystem('reindex_knowledge') re-ingests files")
def test_repair_reindex():
    from tools.executor import repair_subsystem

    result = repair_subsystem(repair="reindex_knowledge")
    assert "OK" in result or "ERROR" in result, f"Should return OK or ERROR: {result}"
    if "OK" in result:
        assert "ingested" in result.lower() or "indexed" in result.lower(), (
            f"Should confirm reindex: {result}"
        )


test_repair_reindex()


@test("repair_subsystem('clear_vram') unloads models")
def test_repair_clear_vram():
    from tools.executor import repair_subsystem

    result = repair_subsystem(repair="clear_vram")
    assert "OK" in result or "ERROR" in result, f"Should return OK or ERROR: {result}"


test_repair_clear_vram()


@test("repair_subsystem rejects invalid repair action")
def test_repair_invalid():
    from tools.executor import repair_subsystem

    result = repair_subsystem(repair="format_disk")
    assert "ERROR" in result and "Unknown repair" in result, (
        f"Should reject invalid repair: {result}"
    )


test_repair_invalid()


@test("repair_subsystem rejects empty repair")
def test_repair_empty():
    from tools.executor import repair_subsystem

    result = repair_subsystem(repair="")
    assert "ERROR" in result, f"Should reject empty repair: {result}"


test_repair_empty()


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
