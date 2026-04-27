"""Full verification suite for deltai training pipeline and all subsystems."""

import asyncio
import os
import sys
import time

os.environ.setdefault("TRAINING_PATH", r"~/deltai/data\training")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

passed = 0
failed = 0


def check(name, condition, info=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}" + (f" ({info})" if info else ""))
    else:
        failed += 1
        print(f"  [FAIL] {name}" + (f" ({info})" if info else ""))


def run():
    global passed, failed

    print("=" * 60)
    print("FULL VERIFICATION SUITE")
    print("=" * 60)

    # === 1. TRAINING MODULE ===
    print("\n--- 1. Training Module ---")
    import training

    training._lora_deps_available = None
    ok, reason = training.check_lora_deps()
    check("LoRA deps available", ok, reason or "READY")

    test_ds = "deltai-verify-full"
    training.delete_dataset(test_ds)
    training.create_dataset(test_ds)
    labels = ["VRAM routing", "RAG pipeline", "GPU protection", "telemetry", "split workload"]
    for i in range(5):
        training.add_example(
            test_ds,
            f"Question {i} about deltai",
            f"deltai uses a robust system for managing {labels[i]} across operational modes.",
            "test",
        )
    ds = training.get_dataset(test_ds)
    check("Dataset CRUD", len(ds["examples"]) == 5, f"{len(ds['examples'])} examples")

    for fmt in ["alpaca", "sharegpt", "chatml"]:
        r = training.export_dataset(test_ds, fmt)
        check(f"Export {fmt}", r["status"] == "ok")

    r = training.auto_capture(
        test_ds,
        "How are my tires?",
        "FL running hot at 102C near degradation cliff. Rears optimal at 88C. Adjust brake bias forward.",
        category="telemetry_coaching",
        rag_context="Lap 15: FL 102C, FR 96C, RL 88C, RR 87C. Medium compound.",
    )
    check("Racing auto-capture with RAG", r["captured"])

    ds = training.get_dataset(test_ds)
    last = ds["examples"][-1]
    check(
        "RAG context in training input", "[Context]" in last["input"] and "[Query]" in last["input"]
    )

    status = training.get_training_status()
    fields = [
        "running",
        "progress",
        "status",
        "loss",
        "mode",
        "adapter_path",
        "gguf_path",
        "trainable_params",
    ]
    fields_ok = all(f in status for f in fields)
    check("Training state fields", fields_ok)

    convert = training._find_convert_script()
    quantize = training._find_quantize_binary()
    check("GGUF convert script", convert is not None, str(convert))
    check("GGUF quantize binary", quantize is not None, str(quantize))

    training.delete_dataset(test_ds)

    # === 2. SAFETY GUARDS ===
    print("\n--- 2. Safety Guards ---")
    import router

    test_ds2 = "deltai-guard-test"
    training.delete_dataset(test_ds2)
    training.create_dataset(test_ds2)
    training.add_example(
        test_ds2,
        "test",
        "Long response for quality filter passing in the training pipeline verification suite.",
        "test",
    )

    # Sim guard
    orig_func = router.is_sim_running
    router.is_sim_running = lambda: True
    r = training.start_training(test_ds2, mode="lora")
    check(
        "LoRA blocked when focus workload active",
        r["status"] == "error" and "focus" in r["reason"].lower(),
        r.get("reason", ""),
    )
    router.is_sim_running = orig_func

    # Concurrency guard
    training._update_state(running=True, status="testing")
    r = training.start_training(test_ds2, mode="fewshot")
    check(
        "Concurrent training rejected",
        r["status"] == "error" and "already in progress" in r["reason"].lower(),
    )
    training._update_state(running=False, status="idle")

    # Cancel mechanism
    training._update_state(running=True, status="testing")
    r = training.stop_training()
    check("stop_training sets cancel flag", r["status"] == "ok")
    training._training_cancel_flag.clear()
    training._update_state(running=False, status="idle")

    training.delete_dataset(test_ds2)

    # === 3. ROUTER + VRAM STRESS ===
    print("\n--- 3. Router Integration + VRAM Stress ---")

    try:
        vram = router.get_vram_free_mb()
        check("VRAM detection", vram > 0, f"{vram}MB free")
    except Exception as e:
        check("VRAM detection", False, str(e))

    # Rapid stress
    start = time.time()
    for i in range(100):
        asyncio.run(router.route(f"stress query {i}"))
    elapsed = (time.time() - start) * 1000
    check("100 rapid routes", elapsed < 10000, f"{elapsed:.0f}ms, {elapsed / 100:.1f}ms/query")

    d = asyncio.run(router.route("What is deltai?"))
    check("Normal routing", d.backend in ("ollama", "anthropic"), f"{d.backend}/{d.model}")

    # === 4. PERSISTENCE ===
    print("\n--- 4. Persistence ---")
    import persistence

    persistence.init_db()
    check("SQLite init", True)

    persistence.save_history_pair("verify user", "verify assistant")
    rows = persistence.load_history(max_turns=10)
    check("History persistence", len(rows) >= 1, f"{len(rows)} rows")

    # === 5. RAG MEMORY ===
    print("\n--- 5. RAG Memory ---")
    import memory

    memory.get_collection()
    check("ChromaDB initialized", memory._collection is not None)

    r = memory.query_knowledge("test query", source_filter="ingest:test", max_age_sec=60)
    check("query_knowledge with filters", isinstance(r, list))

    items = [
        {"source": "test-verify", "context": f"Item {i}", "ttl": 60, "tags": ["test"]}
        for i in range(5)
    ]
    r = memory.ingest_context_batch(items)
    check("Batch ingest", r.get("ingested", 0) == 5, f"{r.get('ingested', 0)} items")

    # === 6. PYTORCH + CUDA ===
    print("\n--- 6. PyTorch + CUDA ---")
    import torch

    check("PyTorch loaded", True, torch.__version__)
    check(
        "CUDA available",
        torch.cuda.is_available(),
        torch.version.cuda if torch.cuda.is_available() else "NO CUDA",
    )

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        check("GPU detected", True, f"{gpu_name} {vram_gb:.1f}GB")

        t = torch.randn(100, 100).cuda()
        result = torch.mm(t, t)
        check("CUDA tensor ops", result.shape == (100, 100))
        del t, result
        torch.cuda.empty_cache()

        import bitsandbytes as bnb

        check("bitsandbytes loaded", True, bnb.__version__)

    # === 7. COMPUTATION DELEGATION TOOLS ===
    print("\n--- 7. Computation Delegation Tools ---")
    from tools.executor import calculate, lookup_reference, summarize_data

    # calculate — basic math
    r = calculate("2 + 2")
    check("calculate basic", "4" in r, r.strip())

    r = calculate("math.sqrt(144)")
    check("calculate math.sqrt", "12" in r, r.strip())

    r = calculate("statistics.mean([10, 20, 30])")
    check("calculate statistics", "20" in r, r.strip())

    # calculate — engineering formula with description
    r = calculate(
        "9.81 * 75 * math.sin(math.radians(30))", "Force component on 30deg incline, 75kg"
    )
    check("calculate with description", "367" in r, r.strip())

    # calculate — security: blocked patterns
    r = calculate("import os")
    check("calculate blocks import", "ERROR" in r or "Blocked" in r, r.strip())

    r = calculate("__builtins__")
    check("calculate blocks __builtins__", "ERROR" in r or "Blocked" in r, r.strip())

    r = calculate("")
    check("calculate empty rejected", "ERROR" in r)

    r = calculate("x" * 600)
    check("calculate length cap", "ERROR" in r and "too long" in r.lower())

    # summarize_data — numeric JSON
    r = summarize_data("[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]")
    check("summarize_data JSON list", "Mean" in r and "5.5" in r, r.split("\n")[0])

    # summarize_data — text with numbers
    r = summarize_data("Lap 1: 1:32.5\nLap 2: 1:31.8\nLap 3: 1:33.1")
    check("summarize_data text extraction", "values" in r.lower() or "Count" in r)

    # summarize_data — empty
    r = summarize_data("")
    check("summarize_data empty", "ERROR" in r)

    # lookup_reference — basic query (depends on ChromaDB having data)
    r = lookup_reference("test query")
    check("lookup_reference callable", isinstance(r, str) and len(r) > 0)

    # lookup_reference — empty query
    r = lookup_reference("")
    check("lookup_reference empty rejected", "ERROR" in r)

    # Tool registration
    from tools.definitions import TOOL_MAP, TOOLS

    tool_names = [t["function"]["name"] for t in TOOLS]
    check("calculate in TOOLS", "calculate" in tool_names)
    check("summarize_data in TOOLS", "summarize_data" in tool_names)
    check("lookup_reference in TOOLS", "lookup_reference" in tool_names)
    check(
        "TOOL_MAP has new tools",
        all(t in TOOL_MAP for t in ["calculate", "summarize_data", "lookup_reference"]),
    )

    from tools.executor import EXECUTORS

    check(
        "EXECUTORS has new tools",
        all(t in EXECUTORS for t in ["calculate", "summarize_data", "lookup_reference"]),
    )

    # === 7b. server_network registry (extension logic) ===
    print("\n--- 7b. Server network registry ---")
    import uuid

    from extensions import server_network
    from extensions.server_network import registry as srv_reg

    _orig_data = os.environ.get("DELTA_DATA_DIR")
    _sn_tmp = os.path.join(
        os.path.expanduser("~"), ".cache", f"deltai-server-network-test-{uuid.uuid4().hex}"
    )
    os.makedirs(_sn_tmp, exist_ok=True)
    try:
        os.environ["DELTA_DATA_DIR"] = _sn_tmp
        p = srv_reg.registry_path()
        check("server_network registry under data dir", p.startswith(os.path.realpath(_sn_tmp)))
        if os.path.isfile(p):
            os.remove(p)
        rec = srv_reg.add_server("10.0.0.1", "testuser", port=2222, label="lab", tags=["a"])
        check("server_network add_server", rec.get("host") == "10.0.0.1" and rec.get("port") == 2222)
        lst = srv_reg.list_servers()
        check("server_network list_servers", len(lst) == 1)
        got = srv_reg.get_server(rec["id"])
        check("server_network get_server", got is not None and got["user"] == "testuser")
        srv_reg.update_server(rec["id"], notes="n1")
        got2 = srv_reg.get_server(rec["id"])
        check("server_network update_server", got2 and got2.get("notes") == "n1")
        ok_del = srv_reg.remove_server(rec["id"])
        check("server_network remove_server", ok_del and srv_reg.list_servers() == [])
        # filter_tools uses core TOOLS; merge extension defs like main does at startup
        from tools.definitions import TOOLS, _merge_extension_tools, filter_tools

        _merge_extension_tools(server_network.TOOLS)
        ft = filter_tools(TOOLS, query="list my homelab servers over ssh")
        fnames = {t["function"]["name"] for t in ft}
        check(
            "filter_tools includes server_network (tier 1)",
            "server_network_list" in fnames and "server_network_probe" in fnames,
        )
        # Handlers accept kwargs like execute_tool (regression)
        out = server_network._list_handler()
        check("server_network_list handler", "registry_path" in out and "servers" in out)
    finally:
        if _orig_data is None:
            os.environ.pop("DELTA_DATA_DIR", None)
        else:
            os.environ["DELTA_DATA_DIR"] = _orig_data
        try:
            rp = os.path.join(_sn_tmp, "local_server_network.json")
            if os.path.isfile(rp):
                os.remove(rp)
            os.rmdir(_sn_tmp)
        except OSError:
            pass

    # === 8. ANTHROPIC CLIENT ===
    print("\n--- 8. Anthropic Client ---")
    import anthropic_client

    check("anthropic_client imports", True)
    has_split = "split_mode" in anthropic_client.stream_chat.__code__.co_varnames
    check("split_mode parameter", has_split)

    # === 9. MCP bridge (optional extra) ===
    print("\n--- 9. MCP bridge ---")
    try:
        import mcp.types  # noqa: F401
        import mcp_bridge

        mcp_bridge.ensure_mcp_tool_catalog()
        mcp_tools = mcp_bridge.ollama_catalog_to_mcp_tools()
        check("mcp ollama_catalog_to_mcp_tools", len(mcp_tools) > 0)
        srv = mcp_bridge.build_mcp_server()
        check("mcp_bridge server builds", srv is not None)
    except ImportError:
        check("mcp extra (skipped)", True)
        print("      (install with: pip install -e '../.[mcp]' from repo root)")

    # === SUMMARY ===
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"RESULTS: {passed}/{total} PASSED, {failed} FAILED")
    if failed == 0:
        print("STATUS: ALL SYSTEMS NOMINAL")
    else:
        print("STATUS: ISSUES DETECTED")
    print("=" * 60)
    return failed


if __name__ == "__main__":
    sys.exit(run())
