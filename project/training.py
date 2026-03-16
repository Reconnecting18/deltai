"""
E3N Training Pipeline — Dataset Management
Handles creation, storage, and export of fine-tuning datasets.
Training data stored as JSONL files in C:\\e3n\\data\\training\\datasets\\.
"""

import os
import json
import time
import logging
import subprocess
import threading

logger = logging.getLogger("e3n.training")

TRAINING_PATH = os.getenv("TRAINING_PATH", r"C:\e3n\data\training")
DATASETS_PATH = os.path.join(TRAINING_PATH, "datasets")
ADAPTERS_PATH = os.path.join(TRAINING_PATH, "adapters")
EXPORTS_PATH = os.path.join(TRAINING_PATH, "exports")

# Ensure directories exist
for _p in [DATASETS_PATH, ADAPTERS_PATH, EXPORTS_PATH]:
    os.makedirs(_p, exist_ok=True)


# ── DATASET MANAGEMENT ────────────────────────────────────────────────

def _dataset_path(name: str) -> str:
    """Get full path for a dataset file. Sanitizes name."""
    safe = "".join(c for c in name if c.isalnum() or c in "-_ ").strip()
    if not safe:
        raise ValueError("Invalid dataset name")
    return os.path.join(DATASETS_PATH, f"{safe}.jsonl")


def list_datasets() -> list[dict]:
    """List all datasets with example counts."""
    datasets = []
    for fname in sorted(os.listdir(DATASETS_PATH)):
        if not fname.endswith(".jsonl"):
            continue
        fpath = os.path.join(DATASETS_PATH, fname)
        name = fname[:-6]  # strip .jsonl
        count = 0
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
        except Exception:
            pass
        size_kb = round(os.path.getsize(fpath) / 1024, 1)
        datasets.append({
            "name": name,
            "examples": count,
            "size_kb": size_kb,
        })
    return datasets


def create_dataset(name: str) -> dict:
    """Create a new empty dataset."""
    path = _dataset_path(name)
    if os.path.exists(path):
        return {"status": "error", "reason": "Dataset already exists"}
    try:
        with open(path, "w", encoding="utf-8") as f:
            pass  # empty file
        return {"status": "ok", "name": name}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


def delete_dataset(name: str) -> dict:
    """Delete a dataset file."""
    path = _dataset_path(name)
    if not os.path.exists(path):
        return {"status": "error", "reason": "Dataset not found"}
    try:
        os.remove(path)
        return {"status": "ok", "name": name}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


def get_dataset(name: str) -> dict:
    """Get all examples from a dataset."""
    path = _dataset_path(name)
    if not os.path.exists(path):
        return {"status": "error", "reason": "Dataset not found", "examples": []}
    examples = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                    ex["_index"] = i
                    examples.append(ex)
                except json.JSONDecodeError:
                    continue
        return {"status": "ok", "name": name, "examples": examples}
    except Exception as e:
        return {"status": "error", "reason": str(e), "examples": []}


def add_example(name: str, input_text: str, output_text: str, category: str = "general") -> dict:
    """Append a training example to a dataset."""
    path = _dataset_path(name)
    if not os.path.exists(path):
        return {"status": "error", "reason": "Dataset not found"}
    if not input_text.strip() or not output_text.strip():
        return {"status": "error", "reason": "Input and output cannot be empty"}
    example = {
        "input": input_text.strip(),
        "output": output_text.strip(),
        "category": category,
        "created": int(time.time()),
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


def remove_example(name: str, index: int) -> dict:
    """Remove an example by line index from a dataset."""
    path = _dataset_path(name)
    if not os.path.exists(path):
        return {"status": "error", "reason": "Dataset not found"}
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if index < 0 or index >= len(lines):
            return {"status": "error", "reason": f"Index {index} out of range"}
        lines.pop(index)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


# ── DATASET EXPORT ─────────────────────────────────────────────────────

def export_dataset(name: str, fmt: str = "alpaca") -> dict:
    """
    Export dataset to training-ready format.
    Formats: alpaca (default), sharegpt, chatml
    """
    result = get_dataset(name)
    if result["status"] != "ok":
        return result

    examples = result["examples"]
    if not examples:
        return {"status": "error", "reason": "Dataset is empty"}

    out_name = f"{name}_{fmt}.json"
    out_path = os.path.join(EXPORTS_PATH, out_name)

    try:
        if fmt == "alpaca":
            # Alpaca format: instruction/input/output
            exported = []
            for ex in examples:
                exported.append({
                    "instruction": ex["input"],
                    "input": "",
                    "output": ex["output"],
                })
        elif fmt == "sharegpt":
            # ShareGPT format: conversations
            exported = []
            for ex in examples:
                exported.append({
                    "conversations": [
                        {"from": "human", "value": ex["input"]},
                        {"from": "gpt", "value": ex["output"]},
                    ]
                })
        elif fmt == "chatml":
            # ChatML format
            exported = []
            for ex in examples:
                exported.append({
                    "messages": [
                        {"role": "user", "content": ex["input"]},
                        {"role": "assistant", "content": ex["output"]},
                    ]
                })
        else:
            return {"status": "error", "reason": f"Unknown format: {fmt}"}

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(exported, f, indent=2, ensure_ascii=False)

        return {
            "status": "ok",
            "path": out_path,
            "format": fmt,
            "examples": len(exported),
        }
    except Exception as e:
        return {"status": "error", "reason": str(e)}


# ── TRAINING STATUS ────────────────────────────────────────────────────

_training_state = {
    "running": False,
    "progress": 0,
    "status": "idle",
    "loss": None,
    "started": None,
    "dataset": None,
    "base_model": None,
    "output_model": None,
    "error": None,
    "completed": None,
    "examples_used": 0,
}

_training_lock = threading.Lock()
_training_process: subprocess.Popen | None = None


def _update_state(**kwargs):
    """Thread-safe update of training state."""
    with _training_lock:
        _training_state.update(kwargs)


def get_training_status() -> dict:
    """Get current training status."""
    with _training_lock:
        return dict(_training_state)


# ── TRAINING EXECUTION ────────────────────────────────────────────────

MODELFILES_PATH = os.path.join(TRAINING_PATH, "modelfiles")
os.makedirs(MODELFILES_PATH, exist_ok=True)

# Canned greeting responses — used by auto_capture to filter out non-trainable exchanges
_CANNED_RESPONSES = {
    "Operational.", "Standing by.", "Online.", "Copy. Out.",
    "Online. Morning, Ethan.", "Online. Evening, Ethan.",
    "Powering down comms. Night, Ethan.",
}


def _build_fewshot_modelfile(base_model: str, examples: list[dict], output_path: str) -> str:
    """
    Build an Ollama Modelfile that bakes few-shot examples into the SYSTEM prompt.
    Since Ollama doesn't expose LoRA training via API, we embed curated examples
    as few-shot demonstrations in the system prompt of a new model variant.
    """
    # Read the base modelfile's SYSTEM prompt if one of our known models
    base_system = ""
    known_modelfiles = {
        "e3n-qwen14b": r"C:\e3n\modelfiles\E3N-qwen14b.modelfile",
        "e3n-qwen3b": r"C:\e3n\modelfiles\E3N-qwen3b.modelfile",
        "e3n-nemo": r"C:\e3n\modelfiles\E3N-nemo.modelfile",
        "e3n": r"C:\e3n\modelfiles\E3N.modelfile",
    }
    modelfile_src = known_modelfiles.get(base_model)
    if modelfile_src and os.path.exists(modelfile_src):
        try:
            with open(modelfile_src, "r", encoding="utf-8") as f:
                content = f.read()
            # Extract SYSTEM block content between triple quotes
            import re
            match = re.search(r'SYSTEM\s+"""(.*?)"""', content, re.DOTALL)
            if match:
                base_system = match.group(1).strip()
        except Exception:
            pass

    if not base_system:
        base_system = "You are E3N, a personal AI assistant. Be direct and precise."

    # Build few-shot block from examples (limit to 20 to keep prompt reasonable)
    fewshot_examples = examples[:20]
    fewshot_block = "\n\nFEW-SHOT EXAMPLES — follow the style and reasoning shown:\n"
    for i, ex in enumerate(fewshot_examples, 1):
        user_text = ex.get("input", "")
        assistant_text = ex.get("output", "")
        fewshot_block += f"\n[Example {i}]\nUser: {user_text}\nAssistant: {assistant_text}\n"

    full_system = base_system + fewshot_block

    # Write the Modelfile
    modelfile_content = f'FROM {base_model}\n\nPARAMETER temperature 0.6\nPARAMETER top_p 0.85\n\nSYSTEM """\n{full_system}\n"""\n'

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    return output_path


def _run_training(dataset_name: str, base_model: str, output_model: str):
    """Background thread that executes the training (model creation) pipeline."""
    global _training_process
    try:
        # Step 1: Export dataset
        _update_state(progress=10, status="exporting dataset")
        export_result = export_dataset(dataset_name, fmt="chatml")
        if export_result.get("status") != "ok":
            _update_state(
                running=False, status="failed",
                error=f"Export failed: {export_result.get('reason', 'unknown')}",
                progress=0,
            )
            return

        examples_count = export_result.get("examples", 0)
        _update_state(progress=20, status="loading examples", examples_used=examples_count)

        # Step 2: Load the raw examples for few-shot embedding
        ds_result = get_dataset(dataset_name)
        if ds_result.get("status") != "ok":
            _update_state(
                running=False, status="failed",
                error="Could not read dataset examples",
                progress=0,
            )
            return

        examples = ds_result["examples"]

        # Step 3: Build Modelfile with few-shot examples
        _update_state(progress=40, status="building modelfile")
        modelfile_path = os.path.join(MODELFILES_PATH, f"{output_model}.modelfile")
        _build_fewshot_modelfile(base_model, examples, modelfile_path)

        # Step 4: Create model via ollama CLI
        _update_state(progress=60, status="creating model via ollama")
        logger.info(f"Training: creating {output_model} from {base_model} with {len(examples)} examples")

        proc = subprocess.Popen(
            ["ollama", "create", output_model, "-f", modelfile_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _training_process = proc

        stdout, stderr = proc.communicate(timeout=300)

        _training_process = None

        if proc.returncode != 0:
            error_msg = stderr.strip() or stdout.strip() or "Unknown ollama error"
            _update_state(
                running=False, status="failed",
                error=f"ollama create failed (rc={proc.returncode}): {error_msg}",
                progress=0,
            )
            logger.error(f"Training failed: {error_msg}")
            return

        # Step 5: Success
        _update_state(
            running=False, progress=100, status="completed",
            completed=int(time.time()), error=None,
        )
        logger.info(f"Training complete: {output_model} created with {len(examples)} few-shot examples")

    except subprocess.TimeoutExpired:
        if _training_process:
            _training_process.kill()
            _training_process = None
        _update_state(
            running=False, status="failed",
            error="Model creation timed out (300s limit)",
            progress=0,
        )
        logger.error("Training timed out")
    except Exception as e:
        _training_process = None
        _update_state(
            running=False, status="failed",
            error=str(e), progress=0,
        )
        logger.error(f"Training error: {e}")


def start_training(
    dataset_name: str,
    base_model: str = "e3n-qwen3b",
    output_model: str | None = None,
) -> dict:
    """
    Start fine-tuning by creating an Ollama model with few-shot examples baked in.
    Runs in a background thread. Returns immediately with status.
    """
    with _training_lock:
        if _training_state["running"]:
            return {"status": "error", "reason": "Training already in progress"}

    if output_model is None:
        output_model = f"{base_model}-ft"

    # Validate dataset exists and has examples
    ds_result = get_dataset(dataset_name)
    if ds_result.get("status") != "ok":
        return {"status": "error", "reason": ds_result.get("reason", "Dataset not found")}
    if not ds_result.get("examples"):
        return {"status": "error", "reason": "Dataset is empty — add examples first"}

    # Reset and start
    _update_state(
        running=True, progress=0, status="starting",
        loss=None, started=int(time.time()), dataset=dataset_name,
        base_model=base_model, output_model=output_model,
        error=None, completed=None, examples_used=0,
    )

    thread = threading.Thread(
        target=_run_training,
        args=(dataset_name, base_model, output_model),
        daemon=True,
    )
    thread.start()

    return {
        "status": "ok",
        "message": f"Training started: {output_model} from {base_model}",
        "dataset": dataset_name,
        "base_model": base_model,
        "output_model": output_model,
    }


def stop_training() -> dict:
    """Cancel training if running."""
    global _training_process
    with _training_lock:
        if not _training_state["running"]:
            return {"status": "error", "reason": "No training in progress"}

    # Kill the subprocess if active
    if _training_process is not None:
        try:
            _training_process.kill()
        except Exception:
            pass
        _training_process = None

    _update_state(
        running=False, status="cancelled",
        progress=0, error=None,
    )
    logger.info("Training cancelled by user")
    return {"status": "ok", "message": "Training cancelled"}


# ── AUTO-CAPTURE ──────────────────────────────────────────────────────

def auto_capture(
    dataset_name: str,
    user_msg: str,
    assistant_msg: str,
    min_quality_len: int = 50,
) -> dict:
    """
    Automatically capture good conversation exchanges for training data.
    Filters out low-quality responses (too short, errors, canned greetings).
    """
    # Skip empty
    if not user_msg or not assistant_msg:
        return {"captured": False, "reason": "empty message"}

    user_msg = user_msg.strip()
    assistant_msg = assistant_msg.strip()

    # Skip if assistant response is too short
    if len(assistant_msg) < min_quality_len:
        return {"captured": False, "reason": f"response too short ({len(assistant_msg)} < {min_quality_len})"}

    # Skip error responses
    error_indicators = ["error:", "exception:", "traceback", "failed to", "i can't", "i cannot"]
    lower_msg = assistant_msg.lower()
    for indicator in error_indicators:
        if lower_msg.startswith(indicator):
            return {"captured": False, "reason": f"error response detected ({indicator})"}

    # Skip canned greeting responses
    if assistant_msg in _CANNED_RESPONSES:
        return {"captured": False, "reason": "canned greeting response"}

    # Skip if user message is very short (likely a greeting)
    if len(user_msg) < 4:
        return {"captured": False, "reason": "user message too short (likely greeting)"}

    # Ensure dataset exists (auto-create if needed)
    ds_path = _dataset_path(dataset_name)
    if not os.path.exists(ds_path):
        create_dataset(dataset_name)

    # Passed all filters — add to dataset
    result = add_example(dataset_name, user_msg, assistant_msg, category="auto-captured")
    if result.get("status") == "ok":
        return {"captured": True, "dataset": dataset_name}
    else:
        return {"captured": False, "reason": result.get("reason", "add_example failed")}
