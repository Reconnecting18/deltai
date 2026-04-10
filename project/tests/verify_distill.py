"""Verification suite for deltai knowledge distillation pipeline."""
import sys, os, time, json, tempfile

os.environ.setdefault('TRAINING_PATH', r'~/deltai/data\training')
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
    print('KNOWLEDGE DISTILLATION VERIFICATION SUITE')
    print('=' * 60)

    import training

    # === 1. DISTILLATION CONFIG ===
    print('\n--- 1. Distillation Config ---')

    check('DISTILL_LR is float', isinstance(training.DISTILL_LR, float),
          f'{training.DISTILL_LR}')
    check('DISTILL_LR <= LORA_LR (gentler)', training.DISTILL_LR <= training.LORA_LR,
          f'{training.DISTILL_LR} <= {training.LORA_LR}')
    check('DISTILL_EPOCHS is int', isinstance(training.DISTILL_EPOCHS, int),
          f'{training.DISTILL_EPOCHS}')
    check('DISTILL_REPLAY_RATIO in [0,1]',
          0 <= training.DISTILL_REPLAY_RATIO <= 1,
          f'{training.DISTILL_REPLAY_RATIO}')
    check('DISTILL_WARMUP_RATIO >= LORA_WARMUP_RATIO',
          training.DISTILL_WARMUP_RATIO >= training.LORA_WARMUP_RATIO,
          f'{training.DISTILL_WARMUP_RATIO} >= {training.LORA_WARMUP_RATIO}')

    # === 2. TEACHER DATA GENERATION ===
    print('\n--- 2. Teacher Data Generation ---')

    # Empty queries returns 0
    result = training.generate_teacher_data([], teacher="local14b")
    check('Empty queries returns ok with 0 generated',
          result.get("status") == "ok" and result.get("generated") == 0)

    # Unknown teacher returns error
    result = training.generate_teacher_data(["test"], teacher="unknown_model")
    check('Unknown teacher returns error',
          result.get("status") == "error")

    # Anthropic teacher without API key returns error
    old_key = os.environ.get("ANTHROPIC_API_KEY", "")
    os.environ["ANTHROPIC_API_KEY"] = ""
    result = training.generate_teacher_data(["test"], teacher="anthropic")
    check('Anthropic teacher without key returns error',
          result.get("status") == "error" and "API_KEY" in result.get("reason", ""))
    if old_key:
        os.environ["ANTHROPIC_API_KEY"] = old_key

    # Quality filter constants exist
    check('Error indicators defined',
          len(training._ERROR_INDICATORS) > 0,
          f'{len(training._ERROR_INDICATORS)} indicators')

    # Dataset auto-creation
    test_teacher_ds = "deltai-verify-teacher-test"
    try:
        path = training._dataset_path(test_teacher_ds)
        if os.path.exists(path):
            os.remove(path)
        # Generate with local14b — may fail if Ollama is down, that's OK
        result = training.generate_teacher_data(
            ["What is stress in materials science?"],
            teacher="local14b",
            dataset_name=test_teacher_ds,
        )
        # Either generated successfully or failed due to Ollama being unreachable
        check('Teacher generation handles Ollama gracefully',
              result.get("status") in ("ok", "error"),
              f'status={result.get("status")}, generated={result.get("generated", "N/A")}')
    finally:
        path = training._dataset_path(test_teacher_ds)
        if os.path.exists(path):
            os.remove(path)

    # === 3. DATASET BLENDING ===
    print('\n--- 3. Dataset Blending ---')

    # Create two test datasets
    ds_a = "deltai-verify-blend-a"
    ds_b = "deltai-verify-blend-b"
    ds_out = "deltai-verify-blend-out"

    try:
        # Setup
        for ds in [ds_a, ds_b, ds_out]:
            path = training._dataset_path(ds)
            if os.path.exists(path):
                os.remove(path)

        training.create_dataset(ds_a)
        training.create_dataset(ds_b)
        for i in range(10):
            training.add_example(ds_a, f"question-a-{i}", f"answer-a-{i}", category="cat-a")
        for i in range(6):
            training.add_example(ds_b, f"question-b-{i}", f"answer-b-{i}", category="cat-b")

        # Blend with 50/50 weights
        result = training.blend_datasets(
            [{"dataset": ds_a, "weight": 0.5}, {"dataset": ds_b, "weight": 0.5}],
            output_name=ds_out,
        )
        check('Blend returns ok', result.get("status") == "ok")
        check('Blend total is reasonable',
              result.get("total", 0) > 0,
              f'total={result.get("total")}')
        check('Blend breakdown has both sources',
              ds_a in result.get("breakdown", {}) and ds_b in result.get("breakdown", {}),
              f'breakdown={result.get("breakdown")}')

        # Verify output dataset exists and has content
        out_result = training.get_dataset(ds_out)
        check('Blended dataset readable',
              out_result.get("status") == "ok" and len(out_result.get("examples", [])) > 0,
              f'{len(out_result.get("examples", []))} examples')

        # Categories preserved
        categories = {ex.get("category") for ex in out_result.get("examples", [])}
        check('Categories preserved in blend',
              "cat-a" in categories and "cat-b" in categories,
              f'categories={categories}')

        # Duplicate output name returns error
        result2 = training.blend_datasets(
            [{"dataset": ds_a, "weight": 1.0}],
            output_name=ds_out,
        )
        check('Duplicate output name returns error',
              result2.get("status") == "error")

        # Nonexistent source handled gracefully
        ds_out2 = "deltai-verify-blend-out2"
        result3 = training.blend_datasets(
            [{"dataset": "nonexistent-ds-xyz", "weight": 1.0}],
            output_name=ds_out2,
        )
        check('Nonexistent source returns error',
              result3.get("status") == "error")

        # max_examples cap
        ds_out3 = "deltai-verify-blend-out3"
        result4 = training.blend_datasets(
            [{"dataset": ds_a, "weight": 1.0, "max_examples": 3}],
            output_name=ds_out3,
        )
        check('max_examples cap respected',
              result4.get("status") == "ok" and result4.get("total", 99) <= 3,
              f'total={result4.get("total")}')

        # Reproducibility with same seed
        ds_out4 = "deltai-verify-blend-out4"
        ds_out5 = "deltai-verify-blend-out5"
        r1 = training.blend_datasets(
            [{"dataset": ds_a, "weight": 0.7}, {"dataset": ds_b, "weight": 0.3}],
            output_name=ds_out4, seed=42,
        )
        r2 = training.blend_datasets(
            [{"dataset": ds_a, "weight": 0.7}, {"dataset": ds_b, "weight": 0.3}],
            output_name=ds_out5, seed=42,
        )
        check('Same seed produces same breakdown',
              r1.get("breakdown") == r2.get("breakdown"),
              f'{r1.get("breakdown")} vs {r2.get("breakdown")}')

    finally:
        for ds in [ds_a, ds_b, ds_out, "deltai-verify-blend-out2",
                    "deltai-verify-blend-out3", "deltai-verify-blend-out4", "deltai-verify-blend-out5"]:
            path = training._dataset_path(ds)
            if os.path.exists(path):
                os.remove(path)

    # === 4. RETENTION VERIFICATION ===
    print('\n--- 4. Retention Verification ---')

    check('Retention baseline queries defined',
          len(training._RETENTION_BASELINE_QUERIES) == 3,
          f'{len(training._RETENTION_BASELINE_QUERIES)} categories')

    total_queries = sum(len(v) for v in training._RETENTION_BASELINE_QUERIES.values())
    check('15 baseline queries total', total_queries == 15, f'{total_queries}')

    # verify_retention function exists and has correct signature
    import inspect
    sig = inspect.signature(training.verify_retention)
    check('verify_retention has correct params',
          'model_name' in sig.parameters and 'baseline_model' in sig.parameters and 'min_pass_rate' in sig.parameters)

    # === 5. DISTILL MODE IN START_TRAINING ===
    print('\n--- 5. Distill Mode Integration ---')

    # Distill mode requires teacher_dataset
    result = training.start_training(
        dataset_name="deltai-auto",
        mode="distill",
        teacher_dataset=None,
    )
    check('Distill mode requires teacher_dataset',
          result.get("status") == "error" and "teacher_dataset" in result.get("reason", ""))

    # Distill mode rejects empty teacher dataset
    empty_ds = "deltai-verify-empty-teacher"
    try:
        training.create_dataset(empty_ds)
        result = training.start_training(
            dataset_name="deltai-auto",
            mode="distill",
            teacher_dataset=empty_ds,
        )
        check('Distill rejects empty teacher dataset',
              result.get("status") == "error" and "empty" in result.get("reason", "").lower())
    finally:
        path = training._dataset_path(empty_ds)
        if os.path.exists(path):
            os.remove(path)

    # Distill mode rejects nonexistent teacher dataset
    result = training.start_training(
        dataset_name="deltai-auto",
        mode="distill",
        teacher_dataset="nonexistent-teacher-xyz",
    )
    check('Distill rejects nonexistent teacher',
          result.get("status") == "error")

    # start_training accepts distill mode parameter
    sig = inspect.signature(training.start_training)
    check('start_training has teacher_dataset param', 'teacher_dataset' in sig.parameters)
    check('start_training has replay_datasets param', 'replay_datasets' in sig.parameters)

    # === 6. _run_lora_training OVERRIDE PARAMS ===
    print('\n--- 6. LoRA Override Parameters ---')

    sig = inspect.signature(training._run_lora_training)
    check('_run_lora_training has lr_override', 'lr_override' in sig.parameters)
    check('_run_lora_training has epochs_override', 'epochs_override' in sig.parameters)
    check('_run_lora_training has warmup_override', 'warmup_override' in sig.parameters)

    # === 7. _run_distill_training EXISTS ===
    print('\n--- 7. Distill Training Function ---')

    check('_run_distill_training exists', hasattr(training, '_run_distill_training'))
    sig = inspect.signature(training._run_distill_training)
    check('_run_distill_training has teacher_dataset param', 'teacher_dataset' in sig.parameters)
    check('_run_distill_training has replay_datasets param', 'replay_datasets' in sig.parameters)
    check('_run_distill_training has replay_ratio param', 'replay_ratio' in sig.parameters)

    # ══════════════════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print(f'  RESULTS: {passed}/{passed + failed} passed, {failed} failed')
    print('=' * 60)

    return failed == 0


if __name__ == '__main__':
    success = run()
    sys.exit(0 if success else 1)
