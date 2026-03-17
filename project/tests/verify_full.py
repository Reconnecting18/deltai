"""Full verification suite for E3N training pipeline and all subsystems."""
import sys, os, time, asyncio

os.environ.setdefault('TRAINING_PATH', r'C:\e3n\data\training')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

passed = 0
failed = 0

def check(name, condition, info=""):
    global passed, failed
    if condition:
        passed += 1
        print(f'  [PASS] {name}' + (f' ({info})' if info else ''))
    else:
        failed += 1
        print(f'  [FAIL] {name}' + (f' ({info})' if info else ''))


def run():
    global passed, failed

    print('=' * 60)
    print('FULL VERIFICATION SUITE')
    print('=' * 60)

    # === 1. TRAINING MODULE ===
    print('\n--- 1. Training Module ---')
    import training

    training._lora_deps_available = None
    ok, reason = training.check_lora_deps()
    check('LoRA deps available', ok, reason or 'READY')

    test_ds = 'e3n-verify-full'
    training.delete_dataset(test_ds)
    training.create_dataset(test_ds)
    labels = ["VRAM routing", "RAG pipeline", "GPU protection", "telemetry", "split workload"]
    for i in range(5):
        training.add_example(
            test_ds,
            f'Question {i} about E3N',
            f'E3N uses a robust system for managing {labels[i]} across operational modes.',
            'test'
        )
    ds = training.get_dataset(test_ds)
    check('Dataset CRUD', len(ds['examples']) == 5, f"{len(ds['examples'])} examples")

    for fmt in ['alpaca', 'sharegpt', 'chatml']:
        r = training.export_dataset(test_ds, fmt)
        check(f'Export {fmt}', r['status'] == 'ok')

    r = training.auto_capture(
        test_ds, 'How are my tires?',
        'FL running hot at 102C near degradation cliff. Rears optimal at 88C. Adjust brake bias forward.',
        category='telemetry_coaching',
        rag_context='Lap 15: FL 102C, FR 96C, RL 88C, RR 87C. Medium compound.'
    )
    check('Racing auto-capture with RAG', r['captured'])

    ds = training.get_dataset(test_ds)
    last = ds['examples'][-1]
    check('RAG context in training input', '[Context]' in last['input'] and '[Query]' in last['input'])

    status = training.get_training_status()
    fields = ['running', 'progress', 'status', 'loss', 'mode', 'adapter_path', 'gguf_path', 'trainable_params']
    fields_ok = all(f in status for f in fields)
    check('Training state fields', fields_ok)

    convert = training._find_convert_script()
    quantize = training._find_quantize_binary()
    check('GGUF convert script', convert is not None, str(convert))
    check('GGUF quantize binary', quantize is not None, str(quantize))

    training.delete_dataset(test_ds)

    # === 2. SAFETY GUARDS ===
    print('\n--- 2. Safety Guards ---')
    import router

    orig_session = router._session_active
    orig_started = router._session_started_at

    test_ds2 = 'e3n-guard-test'
    training.delete_dataset(test_ds2)
    training.create_dataset(test_ds2)
    training.add_example(test_ds2, 'test',
        'Long response for quality filter passing in the training pipeline verification suite.', 'test')

    # Session guard
    router._session_active = True
    router._session_started_at = time.time()
    r = training.start_training(test_ds2, mode='lora')
    check('LoRA blocked during session',
          r['status'] == 'error' and 'session' in r['reason'].lower(), r.get('reason', ''))

    r = training.start_training(test_ds2, mode='fewshot')
    check('Fewshot allowed during session', r['status'] == 'ok')
    for _ in range(30):
        if not training.get_training_status()['running']:
            break
        time.sleep(1)

    router._session_active = False
    router._session_started_at = None

    # Sim guard
    orig_func = router.is_sim_running
    router.is_sim_running = lambda: True
    r = training.start_training(test_ds2, mode='lora')
    check('LoRA blocked when sim running',
          r['status'] == 'error' and 'sim' in r['reason'].lower(), r.get('reason', ''))
    router.is_sim_running = orig_func

    # Concurrency guard
    training._update_state(running=True, status='testing')
    r = training.start_training(test_ds2, mode='fewshot')
    check('Concurrent training rejected',
          r['status'] == 'error' and 'already in progress' in r['reason'].lower())
    training._update_state(running=False, status='idle')

    # Cancel mechanism
    training._update_state(running=True, status='testing')
    r = training.stop_training()
    check('stop_training sets cancel flag', r['status'] == 'ok')
    training._training_cancel_flag.clear()
    training._update_state(running=False, status='idle')

    router._session_active = orig_session
    router._session_started_at = orig_started
    training.delete_dataset(test_ds2)

    # === 3. ROUTER + VRAM STRESS ===
    print('\n--- 3. Router Integration + VRAM Stress ---')

    try:
        vram = router.get_vram_free_mb()
        check('VRAM detection', vram > 0, f'{vram}MB free')
    except Exception as e:
        check('VRAM detection', False, str(e))

    # Session GPU protection
    router._session_active = True
    router._session_started_at = time.time()
    decision = asyncio.run(router.route('How are my tires?'))
    gpu_safe = decision.backend == 'anthropic' or 'cpu' in str(decision.reason).lower()
    check('Session GPU protection', gpu_safe, f'{decision.backend} / {decision.reason}')

    # Telemetry classification
    queries = [
        ('fuel remaining', 'telemetry_lookup'),
        ('tire temps', 'telemetry_lookup'),
        ('why am I slow in sector 3', 'telemetry_coaching'),
        ('should I pit now', 'telemetry_strategy'),
        ('full race analysis', 'telemetry_debrief'),
        ('what is Python', None),
    ]
    classify_pass = sum(1 for q, exp in queries if router.classify_telemetry_category(q) == exp)
    check('Telemetry classification', classify_pass == len(queries), f'{classify_pass}/{len(queries)}')

    router._session_active = False
    router._session_started_at = None

    # Rapid stress
    start = time.time()
    for i in range(100):
        asyncio.run(router.route(f'stress query {i}'))
    elapsed = (time.time() - start) * 1000
    check('100 rapid routes', elapsed < 10000, f'{elapsed:.0f}ms, {elapsed/100:.1f}ms/query')

    d = asyncio.run(router.route('What is E3N?'))
    check('Post-session normal routing', d.backend in ('ollama', 'anthropic'), f'{d.backend}/{d.model}')

    # === 4. PERSISTENCE ===
    print('\n--- 4. Persistence ---')
    import persistence
    persistence.init_db()
    check('SQLite init', True)

    persistence.save_history_pair('verify user', 'verify assistant', session_id='verify-session')
    rows = persistence.load_history(max_turns=10, session_id='verify-session')
    check('Session-aware history', len(rows) >= 1, f'{len(rows)} rows')

    export_path = os.path.join(r'C:\e3n\data\training\sessions', 'verify-session.jsonl')
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    persistence.export_session_history('verify-session', export_path)
    check('Session JSONL export', os.path.exists(export_path))

    # === 5. RAG MEMORY ===
    print('\n--- 5. RAG Memory ---')
    import memory
    memory.get_collection()
    check('ChromaDB initialized', memory._collection is not None)

    r = memory.query_knowledge('test query', source_filter='ingest:test', max_age_sec=60)
    check('query_knowledge with filters', isinstance(r, list))

    items = [{'source': 'test-verify', 'context': f'Item {i}', 'ttl': 60, 'tags': ['test']} for i in range(5)]
    r = memory.ingest_context_batch(items)
    check('Batch ingest', r.get('ingested', 0) == 5, f"{r.get('ingested', 0)} items")

    # === 6. PYTORCH + CUDA ===
    print('\n--- 6. PyTorch + CUDA ---')
    import torch
    check('PyTorch loaded', True, torch.__version__)
    check('CUDA available', torch.cuda.is_available(),
          torch.version.cuda if torch.cuda.is_available() else 'NO CUDA')

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        check('GPU detected', True, f'{gpu_name} {vram_gb:.1f}GB')

        t = torch.randn(100, 100).cuda()
        result = torch.mm(t, t)
        check('CUDA tensor ops', result.shape == (100, 100))
        del t, result
        torch.cuda.empty_cache()

        import bitsandbytes as bnb
        check('bitsandbytes loaded', True, bnb.__version__)

    # === 7. ANTHROPIC CLIENT ===
    print('\n--- 7. Anthropic Client ---')
    import anthropic_client
    check('anthropic_client imports', True)
    has_tm = 'telemetry_mode' in anthropic_client.stream_chat.__code__.co_varnames
    check('telemetry_mode parameter', has_tm)

    # === SUMMARY ===
    print('\n' + '=' * 60)
    total = passed + failed
    print(f'RESULTS: {passed}/{total} PASSED, {failed} FAILED')
    if failed == 0:
        print('STATUS: ALL SYSTEMS NOMINAL')
    else:
        print('STATUS: ISSUES DETECTED')
    print('=' * 60)
    return failed


if __name__ == '__main__':
    sys.exit(run())
