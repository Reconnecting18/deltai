"""
E3N Training Pipeline — Dataset Management + LoRA Fine-Tuning
Handles creation, storage, and export of fine-tuning datasets.
Supports two training modes:
  - "fewshot": Bakes examples into system prompt via Ollama (always available)
  - "lora": Real QLoRA fine-tuning via transformers/peft/trl (requires GPU + deps)
Training data stored as JSONL files in C:\\e3n\\data\\training\\datasets\\.
"""

import os
import json
import time
import re
import gc
import shutil
import logging
import subprocess
import threading

import httpx

logger = logging.getLogger("e3n.training")

TRAINING_PATH = os.getenv("TRAINING_PATH", r"C:\e3n\data\training")
DATASETS_PATH = os.path.join(TRAINING_PATH, "datasets")
ADAPTERS_PATH = os.path.join(TRAINING_PATH, "adapters")
EXPORTS_PATH = os.path.join(TRAINING_PATH, "exports")
MODELFILES_PATH = os.path.join(TRAINING_PATH, "modelfiles")
GGUF_PATH = os.path.join(TRAINING_PATH, "gguf")
CHECKPOINTS_PATH = os.path.join(TRAINING_PATH, "checkpoints")
EVAL_PATH = os.path.join(TRAINING_PATH, "eval")

# Ensure directories exist
for _p in [DATASETS_PATH, ADAPTERS_PATH, EXPORTS_PATH, MODELFILES_PATH,
           GGUF_PATH, CHECKPOINTS_PATH, EVAL_PATH]:
    os.makedirs(_p, exist_ok=True)

# ── LORA HYPERPARAMETERS (from .env) ─────────────────────────────────

LORA_R = int(os.getenv("LORA_R", "16"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))
LORA_EPOCHS = int(os.getenv("LORA_EPOCHS", "3"))
LORA_BATCH_SIZE = int(os.getenv("LORA_BATCH_SIZE", "2"))
LORA_GRAD_ACCUM = int(os.getenv("LORA_GRAD_ACCUM", "4"))
LORA_LR = float(os.getenv("LORA_LR", "2e-4"))
LORA_MAX_SEQ_LEN = int(os.getenv("LORA_MAX_SEQ_LEN", "1024"))
LORA_WARMUP_RATIO = float(os.getenv("LORA_WARMUP_RATIO", "0.05"))
LORA_QUANT_METHOD = os.getenv("LORA_QUANT_METHOD", "Q4_K_M")
HF_BASE_MODEL = os.getenv("HF_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
LLAMA_CPP_PATH = os.getenv("LLAMA_CPP_PATH", r"C:\e3n\tools\llama.cpp")


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
        try:
            size_kb = round(os.path.getsize(fpath) / 1024, 1)
        except OSError:
            size_kb = 0
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
            exported = []
            for ex in examples:
                exported.append({
                    "instruction": ex["input"],
                    "input": "",
                    "output": ex["output"],
                })
        elif fmt == "sharegpt":
            exported = []
            for ex in examples:
                exported.append({
                    "conversations": [
                        {"from": "human", "value": ex["input"]},
                        {"from": "gpt", "value": ex["output"]},
                    ]
                })
        elif fmt == "chatml":
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
    "mode": None,
    "adapter_path": None,
    "gguf_path": None,
    "trainable_params": 0,
}

_training_lock = threading.Lock()
_training_process: subprocess.Popen | None = None
_training_cancel_flag = threading.Event()


def _update_state(**kwargs):
    """Thread-safe update of training state."""
    with _training_lock:
        _training_state.update(kwargs)


def get_training_status() -> dict:
    """Get current training status."""
    with _training_lock:
        return dict(_training_state)


# ── LORA DEPENDENCY CHECK ────────────────────────────────────────────

_lora_deps_available: tuple[bool, str] | None = None


def check_lora_deps() -> tuple[bool, str]:
    """Check if LoRA training dependencies are installed. Cached."""
    global _lora_deps_available
    if _lora_deps_available is not None:
        return _lora_deps_available
    missing = []
    for pkg in ["torch", "transformers", "peft", "trl", "bitsandbytes", "accelerate", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        _lora_deps_available = (False, f"Missing: {', '.join(missing)}")
    else:
        _lora_deps_available = (True, "")
    return _lora_deps_available


# ── VRAM MANAGEMENT ──────────────────────────────────────────────────

def _unload_ollama_models_sync():
    """Synchronously unload all Ollama models from VRAM via keep_alive=0."""
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    try:
        resp = httpx.get(f"{ollama_url}/api/ps", timeout=10)
        if resp.status_code != 200:
            return
        loaded = resp.json().get("models", [])
        for m in loaded:
            model_name = m.get("name", "")
            if model_name:
                httpx.post(f"{ollama_url}/api/generate", json={
                    "model": model_name, "prompt": "", "keep_alive": 0
                }, timeout=10)
                logger.info(f"Unloaded {model_name} from VRAM for training")
    except Exception as e:
        logger.warning(f"Failed to unload Ollama models: {e}")


# ── SYSTEM PROMPT EXTRACTION ────────────────────────────────────────

_KNOWN_MODELFILES = {
    "e3n-qwen14b": r"C:\e3n\modelfiles\E3N-qwen14b.modelfile",
    "e3n-qwen3b": r"C:\e3n\modelfiles\E3N-qwen3b.modelfile",
    "e3n-nemo": r"C:\e3n\modelfiles\E3N-nemo.modelfile",
    "e3n": r"C:\e3n\modelfiles\E3N.modelfile",
}


def _read_system_prompt(ollama_model_name: str) -> str:
    """Extract SYSTEM prompt from an existing Ollama modelfile."""
    path = _KNOWN_MODELFILES.get(ollama_model_name)
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            match = re.search(r'SYSTEM\s+"""(.*?)"""', content, re.DOTALL)
            if match:
                return match.group(1).strip()
        except Exception:
            pass
    return "You are E3N, a personal AI assistant. Be direct and precise."


# ── FEWSHOT TRAINING (LEGACY) ────────────────────────────────────────

# Canned greeting responses — used by auto_capture to filter out non-trainable exchanges
_CANNED_RESPONSES = {
    "Operational.", "Standing by.", "Online.", "Copy. Out.",
    "Online. Morning, Ethan.", "Online. Evening, Ethan.",
    "Powering down comms. Night, Ethan.",
}


def _build_fewshot_modelfile(base_model: str, examples: list[dict], output_path: str) -> str:
    """
    Build an Ollama Modelfile that bakes few-shot examples into the SYSTEM prompt.
    Legacy approach — used when LoRA deps are unavailable.
    """
    base_system = _read_system_prompt(base_model)

    # Build few-shot block from examples (limit to 20 to keep prompt reasonable)
    fewshot_examples = examples[:20]
    fewshot_block = "\n\nFEW-SHOT EXAMPLES — follow the style and reasoning shown:\n"
    for i, ex in enumerate(fewshot_examples, 1):
        user_text = ex.get("input", "")
        assistant_text = ex.get("output", "")
        fewshot_block += f"\n[Example {i}]\nUser: {user_text}\nAssistant: {assistant_text}\n"

    full_system = base_system + fewshot_block

    modelfile_content = (
        f'FROM {base_model}\n\n'
        f'PARAMETER temperature 0.6\n'
        f'PARAMETER top_p 0.85\n\n'
        f'SYSTEM """\n{full_system}\n"""\n'
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    return output_path


def _run_fewshot_training(dataset_name: str, base_model: str, output_model: str):
    """Background thread: legacy few-shot model creation via Ollama."""
    global _training_process
    try:
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

        ds_result = get_dataset(dataset_name)
        if ds_result.get("status") != "ok":
            _update_state(
                running=False, status="failed",
                error="Could not read dataset examples",
                progress=0,
            )
            return

        examples = ds_result["examples"]

        _update_state(progress=40, status="building modelfile")
        modelfile_path = os.path.join(MODELFILES_PATH, f"{output_model}.modelfile")
        _build_fewshot_modelfile(base_model, examples, modelfile_path)

        _update_state(progress=60, status="creating model via ollama")
        logger.info(f"Fewshot training: creating {output_model} from {base_model} with {len(examples)} examples")

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
            logger.error(f"Fewshot training failed: {error_msg}")
            return

        _update_state(
            running=False, progress=100, status="completed",
            completed=int(time.time()), error=None,
        )
        logger.info(f"Fewshot training complete: {output_model} created with {len(examples)} examples")

    except subprocess.TimeoutExpired:
        if _training_process:
            _training_process.kill()
            _training_process = None
        _update_state(
            running=False, status="failed",
            error="Model creation timed out (300s limit)",
            progress=0,
        )
        logger.error("Fewshot training timed out")
    except Exception as e:
        _training_process = None
        _update_state(
            running=False, status="failed",
            error=str(e), progress=0,
        )
        logger.error(f"Fewshot training error: {e}")


# ── LORA TRAINING ────────────────────────────────────────────────────

def _prepare_hf_dataset(dataset_name: str, system_prompt: str, eval_split: float = 0.1):
    """
    Convert E3N JSONL dataset to HuggingFace Dataset for SFTTrainer.
    Returns (train_dataset, eval_dataset) or raises on error.
    Format: ChatML messages list, matching Qwen2.5 expected format.
    """
    from datasets import Dataset

    ds_result = get_dataset(dataset_name)
    if ds_result["status"] != "ok" or not ds_result["examples"]:
        raise ValueError(f"Dataset error: {ds_result.get('reason', 'empty or not found')}")

    examples = ds_result["examples"]
    conversations = []
    for ex in examples:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ex["input"]},
            {"role": "assistant", "content": ex["output"]},
        ]
        conversations.append({"messages": messages})

    dataset = Dataset.from_list(conversations)

    # Split into train/eval if enough examples
    if len(conversations) > 20 and eval_split > 0:
        split = dataset.train_test_split(test_size=eval_split, seed=42)
        return split["train"], split["test"]
    else:
        return dataset, None


def _find_quantize_binary() -> str | None:
    """Find the llama-quantize / llama-cpp quantize binary."""
    candidates = [
        os.path.join(LLAMA_CPP_PATH, "build", "bin", "llama-quantize.exe"),
        os.path.join(LLAMA_CPP_PATH, "build", "bin", "Release", "llama-quantize.exe"),
        os.path.join(LLAMA_CPP_PATH, "llama-quantize.exe"),
        os.path.join(LLAMA_CPP_PATH, "quantize.exe"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return shutil.which("llama-quantize") or shutil.which("quantize")


def _find_convert_script() -> str | None:
    """Find the llama.cpp convert_hf_to_gguf.py script."""
    candidates = [
        os.path.join(LLAMA_CPP_PATH, "convert_hf_to_gguf.py"),
        os.path.join(LLAMA_CPP_PATH, "convert-hf-to-gguf.py"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # Check if llama_cpp python package has it
    try:
        import llama_cpp
        pkg_dir = os.path.dirname(llama_cpp.__file__)
        vendor = os.path.join(pkg_dir, "..", "vendor", "llama.cpp", "convert_hf_to_gguf.py")
        if os.path.exists(vendor):
            return vendor
    except ImportError:
        pass
    return None


def _convert_to_gguf(merged_dir: str, output_gguf: str, quant_method: str = "Q4_K_M"):
    """
    Convert a HuggingFace model directory to quantized GGUF.
    Uses llama.cpp convert_hf_to_gguf.py → f16 GGUF → llama-quantize → quantized GGUF.
    """
    convert_script = _find_convert_script()
    if not convert_script:
        raise RuntimeError(
            "No GGUF conversion tool found. Clone llama.cpp to "
            f"{LLAMA_CPP_PATH} or install llama-cpp-python."
        )

    # Step 1: Convert HF → F16 GGUF
    f16_gguf = output_gguf.replace(".gguf", "-f16.gguf")
    logger.info(f"Converting merged model to F16 GGUF: {f16_gguf}")

    proc = subprocess.run(
        ["python", convert_script, merged_dir,
         "--outfile", f16_gguf, "--outtype", "f16"],
        capture_output=True, text=True, timeout=600
    )
    if proc.returncode != 0:
        raise RuntimeError(f"GGUF conversion failed: {proc.stderr[:500]}")

    # Step 2: Quantize F16 → target quantization
    quantize_bin = _find_quantize_binary()
    if quantize_bin:
        logger.info(f"Quantizing to {quant_method}: {output_gguf}")
        proc = subprocess.run(
            [quantize_bin, f16_gguf, output_gguf, quant_method],
            capture_output=True, text=True, timeout=600
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Quantization failed: {proc.stderr[:500]}")
        # Clean up f16 intermediate
        if os.path.exists(f16_gguf):
            os.remove(f16_gguf)
    else:
        # No quantize binary — rename f16 as final (larger but functional)
        os.rename(f16_gguf, output_gguf)
        logger.warning("llama-quantize not found — using f16 GGUF (larger file size)")


def _register_ollama_model(model_name: str, gguf_path: str, system_prompt: str):
    """Create an Ollama model from a GGUF file. Writes Modelfile, runs `ollama create`."""
    modelfile_path = os.path.join(MODELFILES_PATH, f"{model_name}.modelfile")

    modelfile_content = (
        f'FROM {gguf_path}\n\n'
        f'PARAMETER temperature 0.6\n'
        f'PARAMETER top_p 0.85\n\n'
        f'SYSTEM """\n{system_prompt}\n"""\n'
    )

    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    proc = subprocess.run(
        ["ollama", "create", model_name, "-f", modelfile_path],
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ollama create failed: {proc.stderr[:500]}")
    logger.info(f"Registered {model_name} in Ollama from {gguf_path}")


def _run_lora_training(dataset_name: str, base_model: str, output_model: str):
    """
    Background thread: QLoRA fine-tuning via transformers + peft + trl.

    Pipeline:
      1. Check preconditions (no session/sim active)
      2. Unload Ollama models from VRAM
      3. Load 4-bit quantized base model
      4. Apply LoRA adapter config
      5. Train with SFTTrainer
      6. Save adapter
      7. Merge adapter into base (on CPU)
      8. Export to GGUF (Q4_K_M)
      9. Register in Ollama
    """
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        BitsAndBytesConfig, TrainingArguments,
        TrainerCallback,
    )
    from peft import LoraConfig, get_peft_model, PeftModel
    from trl import SFTTrainer

    adapter_dir = os.path.join(ADAPTERS_PATH, output_model)
    merged_dir = os.path.join(ADAPTERS_PATH, f"{output_model}-merged")
    gguf_file = os.path.join(GGUF_PATH, f"{output_model}.gguf")
    checkpoint_dir = os.path.join(CHECKPOINTS_PATH, output_model)

    try:
        # ── Step 0: Precondition checks ──
        _update_state(progress=2, status="checking preconditions", mode="lora")

        try:
            from router import is_session_active, is_sim_running
            if is_session_active():
                _update_state(running=False, status="failed",
                              error="Cannot train during active racing session", progress=0)
                return
            if is_sim_running():
                _update_state(running=False, status="failed",
                              error="Cannot train while sim is running (VRAM needed)", progress=0)
                return
        except ImportError:
            pass  # router not available, skip sim checks

        # ── Step 1: Unload Ollama models from VRAM ──
        _update_state(progress=5, status="unloading Ollama models from VRAM")
        _unload_ollama_models_sync()
        time.sleep(3)  # let VRAM settle

        # ── Step 2: Read system prompt from modelfile ──
        _update_state(progress=8, status="loading system prompt")
        system_prompt = _read_system_prompt(base_model)

        # ── Step 3: Prepare dataset ──
        _update_state(progress=10, status="preparing dataset")
        train_ds, eval_ds = _prepare_hf_dataset(dataset_name, system_prompt)
        examples_count = len(train_ds)
        _update_state(examples_used=examples_count)
        logger.info(f"LoRA: {examples_count} training examples prepared"
                     + (f", {len(eval_ds)} eval" if eval_ds else ""))

        # ── Step 4: Load tokenizer ──
        _update_state(progress=15, status="loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            HF_BASE_MODEL,
            trust_remote_code=True,
            padding_side="right",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ── Step 5: Load 4-bit quantized model ──
        _update_state(progress=20, status="loading base model (4-bit)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            HF_BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.config.use_cache = False

        # ── Step 6: Apply LoRA ──
        _update_state(progress=30, status="applying LoRA adapter")
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        _update_state(trainable_params=trainable)
        logger.info(f"LoRA: {trainable:,} trainable / {total:,} total "
                     f"({100 * trainable / total:.2f}%)")

        # ── Step 7: Training ──
        _update_state(progress=35, status="training (this takes a while)")

        training_args = TrainingArguments(
            output_dir=checkpoint_dir,
            num_train_epochs=LORA_EPOCHS,
            per_device_train_batch_size=LORA_BATCH_SIZE,
            gradient_accumulation_steps=LORA_GRAD_ACCUM,
            learning_rate=LORA_LR,
            warmup_ratio=LORA_WARMUP_RATIO,
            lr_scheduler_type="cosine",
            logging_steps=1,
            save_strategy="epoch",
            save_total_limit=2,
            bf16=True,
            optim="paged_adamw_8bit",
            max_grad_norm=0.3,
            report_to="none",
            dataloader_pin_memory=False,
        )

        # Progress callback — updates training state with step/loss info
        class ProgressCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if state.max_steps > 0:
                    pct = 35 + int(50 * (state.global_step / state.max_steps))
                    loss_val = logs.get("loss") if logs else None
                    _update_state(
                        progress=min(pct, 85),
                        status=f"training step {state.global_step}/{state.max_steps}",
                        loss=round(loss_val, 4) if loss_val is not None else None,
                    )

        # Cancel/safety callback — checks for user cancel and sim launch
        class SafetyCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if _training_cancel_flag.is_set():
                    control.should_training_stop = True
                    return
                # Abort if sim starts mid-training
                try:
                    from router import is_sim_running
                    if is_sim_running():
                        logger.warning("Sim launched during training — aborting to protect VRAM")
                        control.should_training_stop = True
                        _update_state(status="aborted — sim launched")
                except ImportError:
                    pass

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=LORA_MAX_SEQ_LEN,
            callbacks=[ProgressCallback(), SafetyCallback()],
        )

        train_result = trainer.train()
        logger.info(f"LoRA training done: {train_result.metrics}")

        # Check if cancelled
        if _training_cancel_flag.is_set():
            _update_state(running=False, status="cancelled", progress=0, error=None)
            _training_cancel_flag.clear()
            del model, trainer
            torch.cuda.empty_cache()
            gc.collect()
            return

        # ── Step 8: Save adapter ──
        _update_state(progress=86, status="saving LoRA adapter")
        os.makedirs(adapter_dir, exist_ok=True)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        _update_state(adapter_path=adapter_dir)
        logger.info(f"LoRA adapter saved to {adapter_dir}")

        # ── Step 9: Free VRAM, merge on CPU ──
        _update_state(progress=88, status="merging adapter into base model (CPU)")

        del model, trainer
        torch.cuda.empty_cache()
        gc.collect()

        # Reload base in float16 on CPU for merge
        base_model_fp16 = AutoModelForCausalLM.from_pretrained(
            HF_BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
        )
        merged_model = PeftModel.from_pretrained(base_model_fp16, adapter_dir)
        merged_model = merged_model.merge_and_unload()

        os.makedirs(merged_dir, exist_ok=True)
        merged_model.save_pretrained(merged_dir)
        tokenizer = AutoTokenizer.from_pretrained(HF_BASE_MODEL, trust_remote_code=True)
        tokenizer.save_pretrained(merged_dir)
        logger.info(f"Merged model saved to {merged_dir}")

        del base_model_fp16, merged_model
        gc.collect()

        # ── Step 10: Convert to GGUF ──
        _update_state(progress=92, status="converting to GGUF")
        try:
            _convert_to_gguf(merged_dir, gguf_file, LORA_QUANT_METHOD)
            _update_state(gguf_path=gguf_file)
        except RuntimeError as e:
            logger.warning(f"GGUF conversion failed: {e}. Adapter saved but model not registered in Ollama.")
            _update_state(
                running=False, progress=90, status="partial — adapter saved, GGUF conversion failed",
                error=str(e), completed=int(time.time()),
            )
            return

        # ── Step 11: Register in Ollama ──
        _update_state(progress=96, status="registering in Ollama")
        _register_ollama_model(output_model, gguf_file, system_prompt)

        # ── Step 12: Cleanup merged dir (large, no longer needed) ──
        try:
            shutil.rmtree(merged_dir)
            logger.info(f"Cleaned up merged model dir: {merged_dir}")
        except Exception:
            pass

        # ── Done ──
        _update_state(
            running=False, progress=100, status="completed",
            completed=int(time.time()), error=None,
        )
        logger.info(f"LoRA training complete: {output_model} registered in Ollama")

    except Exception as e:
        logger.error(f"LoRA training failed: {e}", exc_info=True)
        _update_state(running=False, status="failed", error=str(e), progress=0)
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()


# ── TRAINING ENTRY POINT ─────────────────────────────────────────────

def start_training(
    dataset_name: str,
    base_model: str = "e3n-qwen3b",
    output_model: str | None = None,
    mode: str = "auto",
) -> dict:
    """
    Start fine-tuning. Mode selection:
      - "lora": Real QLoRA training (requires GPU + deps)
      - "fewshot": Legacy few-shot embedding (always available)
      - "auto": Try LoRA first, fall back to fewshot
    """
    with _training_lock:
        if _training_state["running"]:
            return {"status": "error", "reason": "Training already in progress"}

    if output_model is None:
        output_model = f"{base_model}-ft"

    # Validate dataset
    ds_result = get_dataset(dataset_name)
    if ds_result.get("status") != "ok":
        return {"status": "error", "reason": ds_result.get("reason", "Dataset not found")}
    if not ds_result.get("examples"):
        return {"status": "error", "reason": "Dataset is empty — add examples first"}

    # Determine actual mode
    actual_mode = mode
    if mode == "auto":
        ok, reason = check_lora_deps()
        actual_mode = "lora" if ok else "fewshot"
    elif mode == "lora":
        ok, reason = check_lora_deps()
        if not ok:
            return {"status": "error", "reason": f"LoRA deps unavailable: {reason}"}

    # Session/sim check for LoRA mode
    if actual_mode == "lora":
        try:
            from router import is_session_active, is_sim_running
            if is_session_active():
                return {"status": "error", "reason": "Cannot train during active racing session"}
            if is_sim_running():
                return {"status": "error", "reason": "Cannot train while sim is running"}
        except ImportError:
            pass

    # Reset state and launch
    _training_cancel_flag.clear()
    _update_state(
        running=True, progress=0, status="starting",
        loss=None, started=int(time.time()), dataset=dataset_name,
        base_model=base_model, output_model=output_model,
        error=None, completed=None, examples_used=0,
        mode=actual_mode, adapter_path=None, gguf_path=None,
        trainable_params=0,
    )

    target = _run_lora_training if actual_mode == "lora" else _run_fewshot_training

    thread = threading.Thread(
        target=target,
        args=(dataset_name, base_model, output_model),
        daemon=True,
    )
    thread.start()

    return {
        "status": "ok",
        "mode": actual_mode,
        "message": f"Training started ({actual_mode}): {output_model} from {base_model}",
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

    # Signal the cancel flag (checked by LoRA SafetyCallback)
    _training_cancel_flag.set()

    # Kill subprocess if active (fewshot mode)
    if _training_process is not None:
        try:
            _training_process.kill()
        except Exception:
            pass
        _training_process = None

    _update_state(running=False, status="cancelled", progress=0, error=None)
    logger.info("Training cancelled by user")
    return {"status": "ok", "message": "Training cancelled"}


# ── A/B EVALUATION ───────────────────────────────────────────────────

def _run_eval_inference(model: str, prompt: str) -> tuple[str, int]:
    """Run a single inference on Ollama and return (response_text, latency_ms)."""
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    start = time.time()
    try:
        resp = httpx.post(f"{ollama_url}/api/chat", json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_predict": 512},
        }, timeout=60)
        elapsed = int((time.time() - start) * 1000)
        if resp.status_code == 200:
            return resp.json().get("message", {}).get("content", ""), elapsed
        return f"[ERROR: HTTP {resp.status_code}]", elapsed
    except Exception as e:
        return f"[ERROR: {e}]", int((time.time() - start) * 1000)


def run_ab_eval(
    model_a: str,
    model_b: str,
    dataset_name: str,
    max_examples: int = 20,
) -> dict:
    """
    Run both models on a held-out eval set and compare.
    Returns structured comparison with latency and response data.
    """
    ds_result = get_dataset(dataset_name)
    if ds_result["status"] != "ok":
        return {"status": "error", "reason": ds_result.get("reason", "Dataset error")}
    if not ds_result.get("examples"):
        return {"status": "error", "reason": "Dataset is empty"}

    examples = ds_result["examples"][:max_examples]
    results = []

    for i, ex in enumerate(examples):
        input_text = ex["input"]
        expected = ex["output"]

        logger.info(f"A/B eval {i + 1}/{len(examples)}: {input_text[:60]}...")

        a_resp, a_time = _run_eval_inference(model_a, input_text)
        b_resp, b_time = _run_eval_inference(model_b, input_text)

        results.append({
            "input": input_text[:200],
            "expected_len": len(expected),
            "model_a": {
                "response_len": len(a_resp),
                "latency_ms": a_time,
                "response": a_resp[:300],
            },
            "model_b": {
                "response_len": len(b_resp),
                "latency_ms": b_time,
                "response": b_resp[:300],
            },
        })

    # Save results
    eval_file = f"ab_{model_a}_vs_{model_b}_{int(time.time())}.json"
    eval_path = os.path.join(EVAL_PATH, eval_file)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_a": model_a,
            "model_b": model_b,
            "dataset": dataset_name,
            "timestamp": int(time.time()),
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    avg_a = sum(r["model_a"]["latency_ms"] for r in results) / len(results) if results else 0
    avg_b = sum(r["model_b"]["latency_ms"] for r in results) / len(results) if results else 0

    return {
        "status": "ok",
        "model_a": model_a,
        "model_b": model_b,
        "examples_tested": len(results),
        "path": eval_path,
        "summary": {
            "avg_latency_a_ms": round(avg_a),
            "avg_latency_b_ms": round(avg_b),
            "avg_response_len_a": round(sum(r["model_a"]["response_len"] for r in results) / len(results)) if results else 0,
            "avg_response_len_b": round(sum(r["model_b"]["response_len"] for r in results) / len(results)) if results else 0,
        },
    }


# ── AUTO-CAPTURE ──────────────────────────────────────────────────────

def auto_capture(
    dataset_name: str,
    user_msg: str,
    assistant_msg: str,
    min_quality_len: int = 50,
    category: str = None,
    rag_context: str = None,
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

    # Determine capture category
    capture_category = category or "auto-captured"

    # For racing exchanges, include RAG context in input
    capture_input = user_msg
    if rag_context and category and category.startswith("telemetry_"):
        capture_input = f"[Context]\n{rag_context}\n\n[Query]\n{user_msg}"

    # Passed all filters — add to dataset
    result = add_example(dataset_name, capture_input, assistant_msg, category=capture_category)
    if result.get("status") == "ok":
        return {"captured": True, "dataset": dataset_name}
    else:
        return {"captured": False, "reason": result.get("reason", "add_example failed")}
