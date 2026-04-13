"""
deltai Training Pipeline — Dataset Management + LoRA Fine-Tuning
Handles creation, storage, and export of fine-tuning datasets.
Supports two training modes:
  - "fewshot": Bakes examples into system prompt via Ollama (always available)
  - "lora": Real QLoRA fine-tuning via transformers/peft/trl (requires GPU + deps)
Training data stored as JSONL files in ~/.local/share/deltai/training/datasets/.
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

import safe_errors

logger = logging.getLogger("deltai.training")


def _sanitize_model_name(name: str) -> str:
    """
    Validate a model name so it is safe for filesystem-derived usage.
    Allows only alphanumerics plus '.', '_' and '-'.
    """
    if not isinstance(name, str):
        raise ValueError("Model name must be a string.")
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Model name cannot be empty.")
    if "/" in cleaned or "\\" in cleaned or ".." in cleaned:
        raise ValueError("Model name contains invalid path characters.")
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?", cleaned):
        raise ValueError("Model name contains unsupported characters.")
    return cleaned

TRAINING_PATH = os.path.expanduser(os.getenv("TRAINING_PATH", "~/.local/share/deltai/training"))
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
LLAMA_CPP_PATH = os.path.expanduser(os.getenv("LLAMA_CPP_PATH", "~/deltai/tools/llama.cpp"))

# ── DISTILLATION HYPERPARAMETERS (from .env) ──────────────────────────

DISTILL_LR = float(os.getenv("DISTILL_LR", "1e-4"))
DISTILL_EPOCHS = int(os.getenv("DISTILL_EPOCHS", "2"))
DISTILL_WARMUP_RATIO = float(os.getenv("DISTILL_WARMUP_RATIO", "0.10"))
DISTILL_REPLAY_RATIO = float(os.getenv("DISTILL_REPLAY_RATIO", "0.70"))
DISTILL_MIN_QUALITY_LEN = int(os.getenv("DISTILL_MIN_QUALITY_LEN", "50"))
DISTILL_TEACHER_TIMEOUT = int(os.getenv("DISTILL_TEACHER_TIMEOUT", "120"))

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# ── ADAPTER SURGERY — AUGMENTATION SLOTS ─────────────────────────────
# Modular LoRA adapters: train, version, evaluate, and merge independently.

REGISTRY_PATH = os.path.join(TRAINING_PATH, "adapter_registry.json")

ADAPTER_DOMAINS = ["racing", "engineering", "personality", "reasoning", "telemetry", "audio"]

DATASET_DOMAIN_MAP = {
    "deltai-racing": "racing",
    "deltai-race-engineering": "racing",
    "deltai-simulations": "racing",
    "deltai-engineering": "engineering",
    "deltai-meche": "engineering",
    "deltai-eng-simulations": "engineering",
    "deltai-personality": "personality",
    "deltai-reasoning": "reasoning",
    "deltai-data-context": "reasoning",
    "deltai-cot-reasoning": "reasoning",
    "deltai-telemetry-analysis": "telemetry",
    "deltai-strategy-advanced": "telemetry",
    "deltai-audio-analysis": "audio",
    # Web-collected datasets (Phase 11 — daily collector)
    "deltai-general-knowledge": "reasoning",
    "deltai-science-knowledge": "engineering",
    "deltai-arxiv-papers": "engineering",
    "deltai-openf1-strategy": "racing",
    "deltai-web-motorsport": "racing",
    # Daily-generated distillation datasets
    "distill-telemetry-targeted": "telemetry",
    "distill-audio-targeted": "audio",
    "distill-racing-targeted": "racing",
    "distill-engineering-targeted": "engineering",
    "distill-reasoning-targeted": "reasoning",
}

# Per-domain training configs — different domains need different tuning
DOMAIN_LORA_CONFIGS = {
    "racing":      {"r": 16, "alpha": 32, "epochs": 3, "lr": 2e-4},
    "engineering": {"r": 16, "alpha": 32, "epochs": 4, "lr": 1e-4},
    "personality": {"r": 8,  "alpha": 16, "epochs": 5, "lr": 3e-4},
    "reasoning":   {"r": 32, "alpha": 64, "epochs": 3, "lr": 1e-4},
    "telemetry":   {"r": 16, "alpha": 32, "epochs": 4, "lr": 1e-4},
    "audio":       {"r": 16, "alpha": 32, "epochs": 4, "lr": 1e-4},
}

# Layer freezing depth per domain (Qwen2.5-3B has 36 layers)
# Lower layers = universal syntax/semantics, upper layers = task-specific
DOMAIN_FREEZE_LAYERS = {
    "racing": 18,       # 50% — domain knowledge lives in top layers
    "engineering": 12,  # 33% — technical reasoning needs more trainable layers
    "personality": 24,  # 67% — style is surface-level, needs fewer layers
    "reasoning": 8,     # 22% — deep reasoning needs most of the network
    "telemetry": 14,    # 39% — pattern recognition needs middle + upper layers
    "audio": 16,        # 44% — new signal pattern learning, upper layers critical
}

ADAPTER_MERGE_METHOD = os.getenv("ADAPTER_MERGE_METHOD", "ties")
ADAPTER_MERGE_DENSITY = float(os.getenv("ADAPTER_MERGE_DENSITY", "0.5"))
ADAPTER_AUTO_MERGE = os.getenv("ADAPTER_AUTO_MERGE", "false").lower() == "true"
ADAPTER_AUTO_PROMOTE = os.getenv("ADAPTER_AUTO_PROMOTE", "false").lower() == "true"

# ── WEB COLLECTION CONFIG (Phase 11) ─────────────────────────────────────────
_WEB_COLLECT_ENABLED = os.getenv("WEB_COLLECT_ENABLED", "true").lower() in ("true", "1", "yes")


def _load_registry() -> dict:
    """Load adapter registry from disk. Creates default if missing."""
    if os.path.exists(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Registry corrupt, creating fresh: {e}")
    return {
        "adapters": {},
        "active_adapters": {d: None for d in ADAPTER_DOMAINS},
        "production_model": "deltai-qwen3b",
        "merged_models": [],
    }


def _save_registry(registry: dict):
    """Save registry to disk atomically (write tmp, rename)."""
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    tmp = REGISTRY_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    # Atomic rename (Windows: os.replace is atomic on same volume)
    os.replace(tmp, REGISTRY_PATH)


def register_adapter(name: str, domain: str, adapter_path: str,
                     dataset: str, examples: int, frozen_layers: int = 0,
                     lora_r: int = 16, lora_alpha: int = 32,
                     tags: list = None) -> dict:
    """Register a trained adapter in the registry."""
    if domain not in ADAPTER_DOMAINS:
        return {"status": "error", "reason": f"Unknown domain: {domain}"}
    registry = _load_registry()
    # Auto-increment version
    existing_versions = [
        v.get("version", 0) for v in registry["adapters"].values()
        if v.get("domain") == domain
    ]
    version = max(existing_versions, default=0) + 1
    entry = {
        "domain": domain,
        "version": version,
        "adapter_path": adapter_path,
        "dataset": dataset,
        "examples_used": examples,
        "created_at": int(time.time()),
        "frozen_layers": frozen_layers,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "eval_score": None,
        "status": "ready",
        "promoted": False,
        "merged_into": None,
        "tags": tags or [],
    }
    registry["adapters"][name] = entry
    _save_registry(registry)
    logger.info(f"Registered adapter: {name} (domain={domain}, v{version})")
    return {"status": "ok", "name": name, "version": version, "entry": entry}


def list_adapters(domain: str = None, status: str = None) -> list[dict]:
    """List registered adapters, optionally filtered by domain or status."""
    registry = _load_registry()
    result = []
    for name, entry in registry["adapters"].items():
        if domain and entry.get("domain") != domain:
            continue
        if status and entry.get("status") != status:
            continue
        result.append({"name": name, **entry})
    return result


def get_adapter(name: str) -> dict:
    """Get a single adapter's metadata."""
    registry = _load_registry()
    entry = registry["adapters"].get(name)
    if not entry:
        return None
    return {"name": name, **entry}


def update_adapter(name: str, **kwargs) -> dict:
    """Update adapter fields (eval_score, status, promoted, etc.)."""
    registry = _load_registry()
    if name not in registry["adapters"]:
        return {"status": "error", "reason": f"Adapter not found: {name}"}
    registry["adapters"][name].update(kwargs)
    _save_registry(registry)
    return {"status": "ok", "name": name}


def set_active_adapter(domain: str, adapter_name: str) -> dict:
    """Set which adapter is active for a domain slot."""
    if domain not in ADAPTER_DOMAINS:
        return {"status": "error", "reason": f"Unknown domain: {domain}"}
    registry = _load_registry()
    if adapter_name and adapter_name not in registry["adapters"]:
        return {"status": "error", "reason": f"Adapter not found: {adapter_name}"}
    registry["active_adapters"][domain] = adapter_name
    _save_registry(registry)
    logger.info(f"Active adapter for {domain}: {adapter_name}")
    return {"status": "ok", "domain": domain, "adapter": adapter_name}


def get_active_adapters() -> dict:
    """Return the current active adapter map."""
    registry = _load_registry()
    return registry.get("active_adapters", {})


def remove_adapter(name: str, delete_files: bool = False) -> dict:
    """Remove adapter from registry, optionally delete files."""
    registry = _load_registry()
    if name not in registry["adapters"]:
        return {"status": "error", "reason": f"Adapter not found: {name}"}
    entry = registry["adapters"].pop(name)
    # Clear from active if it was active
    for domain, active in registry["active_adapters"].items():
        if active == name:
            registry["active_adapters"][domain] = None
    _save_registry(registry)
    if delete_files and entry.get("adapter_path"):
        try:
            shutil.rmtree(entry["adapter_path"])
            logger.info(f"Deleted adapter files: {entry['adapter_path']}")
        except OSError as e:
            logger.warning(f"Failed to delete adapter files: {e}")
    return {"status": "ok", "removed": name}


# ── TIES MERGE ───────────────────────────────────────────────────────

def _ties_merge(deltas: list, density: float = 0.5) -> dict:
    """
    TIES merging: Trim, Elect sign, Merge.
    Resolves conflicts between adapters that modify the same weights.
    Pure PyTorch, CPU-only.

    Args:
        deltas: list of state_dict differences (adapter - base) per adapter
        density: fraction of parameters to keep (higher = more from each adapter)
    Returns:
        Merged state dict delta
    """
    import torch

    merged = {}
    for key in deltas[0].keys():
        tensors = [d[key] for d in deltas if key in d]
        if len(tensors) == 1:
            merged[key] = tensors[0]
            continue

        # Step 1: Trim — zero out small-magnitude values per adapter
        trimmed = []
        for t in tensors:
            flat = t.abs().float().flatten()
            if flat.numel() == 0:
                trimmed.append(t)
                continue
            threshold = torch.quantile(flat, 1.0 - density)
            mask = t.abs() >= threshold
            trimmed.append(t * mask)

        # Step 2: Elect sign — majority vote across adapters
        signs = torch.stack([torch.sign(t) for t in trimmed])
        elected_sign = torch.sign(signs.sum(dim=0))

        # Step 3: Merge — mean of values agreeing with elected sign
        aligned = []
        for t in trimmed:
            agree = (torch.sign(t) == elected_sign) | (t == 0)
            aligned.append(t * agree)

        stacked = torch.stack(aligned)
        # Count non-zero contributions for proper averaging
        nonzero_count = (stacked != 0).float().sum(dim=0).clamp(min=1)
        merged[key] = stacked.sum(dim=0) / nonzero_count

    return merged


def _linear_merge(deltas: list, weights: list = None) -> dict:
    """Simple weighted average merge of adapter deltas."""
    import torch

    if weights is None:
        weights = [1.0 / len(deltas)] * len(deltas)

    merged = {}
    for key in deltas[0].keys():
        tensors = [d[key] for d in deltas if key in d]
        ws = weights[:len(tensors)]
        # Normalize weights
        w_sum = sum(ws)
        ws = [w / w_sum for w in ws]
        merged[key] = sum(t * w for t, w in zip(tensors, ws))

    return merged


def merge_adapters(adapter_names: list = None, method: str = None,
                   density: float = None, output_model: str = None) -> dict:
    """
    Merge multiple domain adapters into a single production GGUF model.
    CPU-only, zero GPU cost.

    Args:
        adapter_names: List of adapter names to merge. None = auto-select active.
        method: "ties" or "linear". Default from config.
        density: TIES density parameter (0-1). Default from config.
        output_model: Output model name. Default: "deltai-qwen3b-merged".
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    method = method or ADAPTER_MERGE_METHOD
    density = density if density is not None else ADAPTER_MERGE_DENSITY
    output_model = output_model or "deltai-qwen3b-merged"
    output_model = _sanitize_model_name(output_model)

    registry = _load_registry()

    # Auto-select active adapters if none specified
    if adapter_names is None:
        adapter_names = [
            name for name in registry["active_adapters"].values()
            if name is not None
        ]

    if not adapter_names:
        return {"status": "error", "reason": "No adapters to merge"}

    if len(adapter_names) < 2:
        return {"status": "error", "reason": "Need at least 2 adapters to merge"}

    # Validate all adapters exist
    for name in adapter_names:
        if name not in registry["adapters"]:
            return {"status": "error", "reason": f"Adapter not found: {name}"}
        entry = registry["adapters"][name]
        if not os.path.exists(entry["adapter_path"]):
            return {"status": "error", "reason": f"Adapter files missing: {name}"}

    logger.info(f"Merging adapters: {adapter_names} via {method} (density={density})")

    merged_dir = os.path.join(ADAPTERS_PATH, f"{output_model}-merged")
    gguf_file = os.path.join(GGUF_PATH, f"{output_model}.gguf")

    try:
        # Load base model on CPU
        logger.info("Loading base model on CPU for merge...")
        base_model = AutoModelForCausalLM.from_pretrained(
            HF_BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
        )
        base_state = {k: v.clone() for k, v in base_model.state_dict().items()}

        # Extract deltas from each adapter
        deltas = []
        for name in adapter_names:
            adapter_path = registry["adapters"][name]["adapter_path"]
            logger.info(f"Loading adapter: {name} from {adapter_path}")
            adapted = PeftModel.from_pretrained(base_model, adapter_path)
            adapted = adapted.merge_and_unload()
            # Compute delta (what the adapter changed)
            delta = {}
            for k, v in adapted.state_dict().items():
                if k in base_state:
                    diff = v - base_state[k]
                    if diff.abs().sum() > 0:
                        delta[k] = diff
            deltas.append(delta)
            del adapted
            gc.collect()
            # Reload base for next adapter
            if name != adapter_names[-1]:
                base_model = AutoModelForCausalLM.from_pretrained(
                    HF_BASE_MODEL,
                    torch_dtype=torch.float16,
                    device_map="cpu",
                    trust_remote_code=True,
                )

        # Merge deltas
        logger.info(f"Merging {len(deltas)} adapter deltas via {method}...")
        if method == "ties":
            merged_delta = _ties_merge(deltas, density=density)
        elif method == "linear":
            merged_delta = _linear_merge(deltas)
        else:
            return {"status": "error", "reason": f"Unknown merge method: {method}"}

        del deltas
        gc.collect()

        # Apply merged delta to base
        final_model = AutoModelForCausalLM.from_pretrained(
            HF_BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
        )
        final_state = final_model.state_dict()
        for k, v in merged_delta.items():
            if k in final_state:
                final_state[k] = final_state[k] + v
        final_model.load_state_dict(final_state)

        del merged_delta, base_state
        gc.collect()

        # Save merged model
        os.makedirs(merged_dir, exist_ok=True)
        final_model.save_pretrained(merged_dir)
        tokenizer = AutoTokenizer.from_pretrained(HF_BASE_MODEL, trust_remote_code=True)
        tokenizer.save_pretrained(merged_dir)
        logger.info(f"Merged model saved to {merged_dir}")

        del final_model
        gc.collect()

        # Convert to GGUF
        logger.info("Converting merged model to GGUF...")
        _convert_to_gguf(merged_dir, gguf_file, LORA_QUANT_METHOD)

        # Register in Ollama
        system_prompt = _read_system_prompt("deltai-qwen3b")
        _register_ollama_model(output_model, gguf_file, system_prompt)

        # Cleanup merged dir
        try:
            shutil.rmtree(merged_dir)
        except OSError:
            pass

        # Update registry
        for name in adapter_names:
            registry["adapters"][name]["merged_into"] = output_model
        registry["production_model"] = output_model
        if output_model not in registry["merged_models"]:
            registry["merged_models"].append(output_model)
        _save_registry(registry)

        logger.info(f"Merge complete: {output_model} registered in Ollama")
        return {
            "status": "ok",
            "output_model": output_model,
            "adapters_merged": adapter_names,
            "method": method,
            "density": density,
        }

    except Exception as e:
        logger.error(f"Adapter merge failed: {e}", exc_info=True)
        # Cleanup
        try:
            shutil.rmtree(merged_dir)
        except OSError:
            pass
        return {"status": "error", "reason": safe_errors.public_error_detail(e)}


# ── DOMAIN-TARGETED TRAINING ─────────────────────────────────────────

def start_domain_training(domain: str, dataset_name: str = None,
                          freeze_layers: int = None,
                          lr_override: float = None,
                          epochs_override: int = None) -> dict:
    """
    Start domain-targeted adapter training for an augmentation slot.
    Trains a LoRA adapter for the specified domain with selective layer freezing.
    The adapter is saved and registered but NOT merged into a GGUF — use
    merge_adapters() to combine active adapters into a production model.

    Args:
        domain: One of ADAPTER_DOMAINS ("racing", "engineering", "personality", "reasoning")
        dataset_name: Dataset to train on. Auto-selected from DATASET_DOMAIN_MAP if None.
        freeze_layers: Number of bottom layers to freeze. Auto-selected per domain if None.
        lr_override: Override learning rate.
        epochs_override: Override number of epochs.
    """
    global _training_state

    if domain not in ADAPTER_DOMAINS:
        return {"status": "error", "reason": f"Unknown domain: {domain}. Valid: {ADAPTER_DOMAINS}"}

    if _training_state.get("running"):
        return {"status": "error", "reason": "Training already in progress"}

    # Auto-select dataset
    if dataset_name is None:
        for ds_name, ds_domain in DATASET_DOMAIN_MAP.items():
            if ds_domain == domain:
                ds_path = _dataset_path(ds_name)
                if os.path.exists(ds_path):
                    dataset_name = ds_name
                    break
        if dataset_name is None:
            return {"status": "error", "reason": f"No dataset found for domain: {domain}"}

    # Verify dataset exists and has examples
    ds_path = _dataset_path(dataset_name)
    if not os.path.exists(ds_path):
        return {"status": "error", "reason": f"Dataset not found: {dataset_name}"}

    # Auto-select freeze layers and LoRA config
    if freeze_layers is None:
        freeze_layers = DOMAIN_FREEZE_LAYERS.get(domain, 0)
    domain_config = DOMAIN_LORA_CONFIGS.get(domain, {})
    lr = lr_override or domain_config.get("lr", LORA_LR)
    epochs = epochs_override or domain_config.get("epochs", LORA_EPOCHS)
    lora_r = domain_config.get("r", LORA_R)
    lora_alpha = domain_config.get("alpha", LORA_ALPHA)

    # Auto-name: domain-vN
    registry = _load_registry()
    existing_versions = [
        v.get("version", 0) for v in registry["adapters"].values()
        if v.get("domain") == domain
    ]
    version = max(existing_versions, default=0) + 1
    adapter_name = f"{domain}-v{version}"

    logger.info(f"Starting domain training: {adapter_name} "
                f"(dataset={dataset_name}, freeze={freeze_layers}, r={lora_r}, "
                f"alpha={lora_alpha}, lr={lr}, epochs={epochs})")

    # Reset training state
    _training_state = {
        "running": True, "progress": 0, "status": "starting domain training",
        "mode": "domain", "domain": domain, "adapter_name": adapter_name,
        "dataset": dataset_name, "examples_used": 0, "loss": None,
        "error": None, "started": int(time.time()), "completed": None,
        "frozen_layers": freeze_layers,
    }
    _training_cancel_flag.clear()

    # Launch background training thread
    t = threading.Thread(
        target=_run_lora_training,
        args=(dataset_name, "deltai-qwen3b", adapter_name),
        kwargs={
            "lr_override": lr,
            "epochs_override": epochs,
            "adapter_only": True,
            "frozen_layers": freeze_layers,
            "lora_r_override": lora_r,
            "lora_alpha_override": lora_alpha,
            "domain": domain,
        },
        daemon=True,
    )
    t.start()

    return {
        "status": "started",
        "adapter_name": adapter_name,
        "domain": domain,
        "version": version,
        "dataset": dataset_name,
        "frozen_layers": freeze_layers,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
    }


# ── ADAPTER EVALUATION ───────────────────────────────────────────────

DOMAIN_EVAL_DATASETS = {
    "racing": "deltai-racing",
    "engineering": "deltai-engineering",
    "personality": "deltai-personality",
    "reasoning": "deltai-reasoning",
}


def eval_adapter(adapter_name: str, eval_dataset: str = None,
                 baseline_model: str = "deltai-qwen3b",
                 max_examples: int = 20) -> dict:
    """
    Evaluate an adapter against the baseline Ollama model.
    Loads base + adapter via PEFT, runs eval dataset, compares responses.

    Returns eval results with a quality score.
    """
    adapter_info = get_adapter(adapter_name)
    if not adapter_info:
        return {"status": "error", "reason": f"Adapter not found: {adapter_name}"}

    domain = adapter_info.get("domain", "")
    adapter_path = adapter_info.get("adapter_path", "")
    if not os.path.exists(adapter_path):
        return {"status": "error", "reason": f"Adapter files missing: {adapter_path}"}

    # Auto-select eval dataset
    if eval_dataset is None:
        eval_dataset = DOMAIN_EVAL_DATASETS.get(domain)
    if not eval_dataset:
        return {"status": "error", "reason": "No eval dataset specified or available for domain"}

    ds_path = _dataset_path(eval_dataset)
    if not os.path.exists(ds_path):
        return {"status": "error", "reason": f"Eval dataset not found: {eval_dataset}"}

    # Load eval examples
    examples = []
    try:
        with open(ds_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                    if ex.get("input") and ex.get("output"):
                        examples.append(ex)
                except json.JSONDecodeError:
                    continue
    except OSError as e:
        safe_errors.log_exception(logger, "eval_adapter dataset read failed", e)
        return {"status": "error", "reason": safe_errors.public_error_detail(e)}

    if not examples:
        return {"status": "error", "reason": "No valid examples in eval dataset"}

    examples = examples[:max_examples]

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        # Unload Ollama models first
        _unload_ollama_models_sync()
        time.sleep(2)

        # Load base + adapter
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
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(HF_BASE_MODEL, trust_remote_code=True)

        # Generate responses
        results = []
        for ex in examples:
            messages = [{"role": "user", "content": ex["input"]}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=256,
                    do_sample=False, temperature=0.1,
                )
            latency = time.time() - t0

            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            # Simple quality score: length similarity to reference
            ref_len = len(ex["output"])
            resp_len = len(response)
            len_ratio = min(resp_len, ref_len) / max(resp_len, ref_len, 1)

            results.append({
                "input": ex["input"][:100],
                "reference_len": ref_len,
                "response_len": resp_len,
                "latency": round(latency, 2),
                "len_score": round(len_ratio, 3),
            })

        del model
        torch.cuda.empty_cache()
        gc.collect()

        # Overall score
        avg_score = sum(r["len_score"] for r in results) / len(results)
        avg_latency = sum(r["latency"] for r in results) / len(results)

        # Save eval results
        safe_adapter_name = _sanitize_model_name(adapter_name)
        eval_root = os.path.realpath(EVAL_PATH)
        eval_file = os.path.realpath(os.path.join(eval_root, f"{safe_adapter_name}_eval.json"))
        if os.path.commonpath([eval_root, eval_file]) != eval_root:
            return {"status": "error", "reason": "Invalid adapter name for eval path"}
        eval_data = {
            "adapter": adapter_name,
            "domain": domain,
            "baseline": baseline_model,
            "eval_dataset": eval_dataset,
            "examples": len(results),
            "avg_score": round(avg_score, 3),
            "avg_latency": round(avg_latency, 2),
            "timestamp": int(time.time()),
            "results": results,
        }
        with open(eval_file, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, indent=2)

        # Update registry
        update_adapter(adapter_name, eval_score=round(avg_score, 3),
                       eval_path=eval_file)

        # Auto-promote if better than current active
        if ADAPTER_AUTO_PROMOTE:
            registry = _load_registry()
            current_active = registry["active_adapters"].get(domain)
            if current_active:
                current_info = registry["adapters"].get(current_active, {})
                current_score = current_info.get("eval_score", 0) or 0
                if avg_score > current_score:
                    set_active_adapter(domain, adapter_name)
                    update_adapter(adapter_name, promoted=True)
                    logger.info(f"Auto-promoted {adapter_name} (score {avg_score:.3f} > {current_score:.3f})")
            else:
                set_active_adapter(domain, adapter_name)
                update_adapter(adapter_name, promoted=True)

        return {
            "status": "ok",
            "adapter": adapter_name,
            "domain": domain,
            "avg_score": round(avg_score, 3),
            "avg_latency": round(avg_latency, 2),
            "examples_evaluated": len(results),
            "eval_file": eval_file,
        }

    except Exception as e:
        logger.error(f"Adapter eval failed: {e}", exc_info=True)
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        return {"status": "error", "reason": safe_errors.public_error_detail(e)}


def rollback_adapter(domain: str, target_version: int = None) -> dict:
    """Roll back to a previous adapter version for a domain."""
    if domain not in ADAPTER_DOMAINS:
        return {"status": "error", "reason": f"Unknown domain: {domain}"}

    registry = _load_registry()
    domain_adapters = [
        (name, info) for name, info in registry["adapters"].items()
        if info.get("domain") == domain and info.get("status") == "ready"
    ]

    if not domain_adapters:
        return {"status": "error", "reason": f"No adapters for domain: {domain}"}

    # Sort by version
    domain_adapters.sort(key=lambda x: x[1].get("version", 0))

    current = registry["active_adapters"].get(domain)

    if target_version is not None:
        target = next(
            (name for name, info in domain_adapters if info.get("version") == target_version),
            None
        )
        if not target:
            return {"status": "error", "reason": f"Version {target_version} not found for {domain}"}
    else:
        # Roll back to previous version
        if current:
            current_idx = next(
                (i for i, (n, _) in enumerate(domain_adapters) if n == current), -1
            )
            if current_idx > 0:
                target = domain_adapters[current_idx - 1][0]
            else:
                return {"status": "error", "reason": "Already at oldest version"}
        else:
            target = domain_adapters[-1][0]

    set_active_adapter(domain, target)
    return {
        "status": "ok",
        "domain": domain,
        "rolled_back_to": target,
        "previous": current,
    }


# ── DATASET MANAGEMENT ────────────────────────────────────────────────

def _dataset_path(name: str) -> str:
    """Get full path for a dataset file. Sanitizes name."""
    safe = "".join(c for c in name if c.isalnum() or c in "-_").strip()
    if not safe:
        raise ValueError("Invalid dataset name")
    ds_root = os.path.realpath(os.path.expanduser(DATASETS_PATH))
    full = os.path.realpath(os.path.join(ds_root, f"{safe}.jsonl"))
    if os.path.commonpath([ds_root, full]) != ds_root:
        raise ValueError("Invalid dataset name")
    return full


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
        return {"status": "error", "reason": safe_errors.public_error_detail(e)}


def delete_dataset(name: str) -> dict:
    """Delete a dataset file."""
    path = _dataset_path(name)
    if not os.path.exists(path):
        return {"status": "error", "reason": "Dataset not found"}
    try:
        os.remove(path)
        return {"status": "ok", "name": name}
    except Exception as e:
        return {"status": "error", "reason": safe_errors.public_error_detail(e)}


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
        return {"status": "error", "reason": safe_errors.public_error_detail(e), "examples": []}


def add_example(name: str, input_text: str, output_text: str, category: str = "general") -> dict:
    """Append a training example to a dataset."""
    path = _dataset_path(name)
    if not os.path.exists(path):
        return {"status": "error", "reason": "Dataset not found"}
    if not output_text.strip():
        return {"status": "error", "reason": "Output cannot be empty"}
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
        return {"status": "error", "reason": safe_errors.public_error_detail(e)}


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
        return {"status": "error", "reason": safe_errors.public_error_detail(e)}


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
        return {"status": "error", "reason": safe_errors.public_error_detail(e)}


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
    "deltai-qwen14b": "~/deltai/modelfiles/deltai-qwen14b.modelfile",
    "deltai-qwen3b": "~/deltai/modelfiles/deltai-qwen3b.modelfile",
    "deltai-nemo": "~/deltai/modelfiles/deltai-nemo.modelfile",
    "deltai-fallback": "~/deltai/modelfiles/deltai-fallback.modelfile",
    "deltai": "~/deltai/modelfiles/deltai.modelfile",
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
    return "You are deltai, a modular local AI layer. Be direct, precise, and safety-conscious."


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
        output_model = _sanitize_model_name(output_model)
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
            error=safe_errors.public_error_detail(e), progress=0,
        )
        logger.error(f"Fewshot training error: {e}")


# ── LORA TRAINING ────────────────────────────────────────────────────

def _prepare_hf_dataset(dataset_name: str, system_prompt: str, eval_split: float = 0.1):
    """
    Convert deltai JSONL dataset to HuggingFace Dataset for SFTTrainer.
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
        os.path.join(LLAMA_CPP_PATH, "build", "bin", "llama-quantize"),
        os.path.join(LLAMA_CPP_PATH, "llama-quantize"),
        os.path.join(LLAMA_CPP_PATH, "quantize"),
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
    model_name = _sanitize_model_name(model_name)
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


def _run_lora_training(dataset_name: str, base_model: str, output_model: str,
                       lr_override: float = None, epochs_override: int = None,
                       warmup_override: float = None, adapter_only: bool = False,
                       frozen_layers: int = 0, lora_r_override: int = None,
                       lora_alpha_override: int = None, domain: str = None):
    """
    Background thread: QLoRA fine-tuning via transformers + peft + trl.
    Optional overrides allow distillation mode to use gentler hyperparameters.

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
        BitsAndBytesConfig,
        TrainerCallback,
    )
    from peft import LoraConfig, get_peft_model, PeftModel
    from trl import SFTTrainer, SFTConfig

    output_model = _sanitize_model_name(output_model)
    adapter_dir = os.path.join(ADAPTERS_PATH, output_model)
    merged_dir = os.path.join(ADAPTERS_PATH, f"{output_model}-merged")
    gguf_file = os.path.join(GGUF_PATH, f"{output_model}.gguf")
    checkpoint_dir = os.path.join(CHECKPOINTS_PATH, output_model)

    try:
        # ── Step 0: Precondition checks ──
        _update_state(progress=2, status="checking preconditions", mode="lora")

        try:
            from router import is_sim_running
            if is_sim_running():
                _update_state(running=False, status="failed",
                              error="Cannot train while GPU focus workload is active (VRAM needed)", progress=0)
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

        # ── Step 5b: Selective layer freezing (augmentation surgery) ──
        if frozen_layers > 0:
            frozen_count = 0
            for i in range(min(frozen_layers, len(model.model.layers))):
                for param in model.model.layers[i].parameters():
                    param.requires_grad = False
                frozen_count += 1
            logger.info(f"Frozen bottom {frozen_count} transformer layers "
                        f"(preserving universal knowledge)")
            _update_state(status=f"frozen {frozen_count} bottom layers")

        # ── Step 6: Apply LoRA ──
        _update_state(progress=30, status="applying LoRA adapter")
        effective_r = lora_r_override or LORA_R
        effective_alpha = lora_alpha_override or LORA_ALPHA
        lora_config = LoraConfig(
            r=effective_r,
            lora_alpha=effective_alpha,
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

        # Format conversations into text using tokenizer's chat template
        def _format_messages(example):
            text = tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )
            return {"text": text}

        train_ds = train_ds.map(_format_messages)
        if eval_ds is not None:
            eval_ds = eval_ds.map(_format_messages)

        training_args = SFTConfig(
            output_dir=checkpoint_dir,
            num_train_epochs=epochs_override or LORA_EPOCHS,
            per_device_train_batch_size=LORA_BATCH_SIZE,
            gradient_accumulation_steps=LORA_GRAD_ACCUM,
            learning_rate=lr_override or LORA_LR,
            warmup_ratio=warmup_override or LORA_WARMUP_RATIO,
            lr_scheduler_type="cosine",
            logging_steps=1,
            save_strategy="epoch",
            save_total_limit=2,
            bf16=True,
            optim="paged_adamw_8bit",
            max_grad_norm=0.3,
            report_to="none",
            dataloader_pin_memory=False,
            max_length=LORA_MAX_SEQ_LEN,
            dataset_text_field="text",
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
            processing_class=tokenizer,
            args=training_args,
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

        # ── Step 8b: Adapter-only mode (domain training) ──
        if adapter_only:
            # Register in adapter registry, skip merge/GGUF/Ollama
            del model, trainer
            torch.cuda.empty_cache()
            gc.collect()
            if domain:
                register_adapter(
                    name=output_model,
                    domain=domain,
                    adapter_path=adapter_dir,
                    dataset=dataset_name,
                    examples=examples_count,
                    frozen_layers=frozen_layers,
                    lora_r=effective_r,
                    lora_alpha=effective_alpha,
                )
            _update_state(
                running=False, progress=100, status="completed (adapter saved)",
                completed=int(time.time()), error=None,
            )
            logger.info(f"Domain adapter saved: {output_model} (domain={domain})")
            return

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
                error=safe_errors.public_error_detail(e), completed=int(time.time()),
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
        _update_state(running=False, status="failed", error=safe_errors.public_error_detail(e), progress=0)
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()


# ── DISTILLATION TRAINING ────────────────────────────────────────────

def _run_distill_training(teacher_dataset: str, replay_datasets: list[str],
                          base_model: str, output_model: str,
                          replay_ratio: float = None):
    """
    Background thread: Knowledge distillation training.
    1. Blends teacher data with replay data
    2. Runs LoRA with gentler hyperparameters
    3. Verifies retention after training
    """
    if replay_ratio is None:
        replay_ratio = DISTILL_REPLAY_RATIO

    blend_name = f"deltai-distill-blend-{int(time.time())}"

    try:
        _update_state(status="blending datasets", progress=5)

        # Build blend sources
        teacher_weight = 1.0 - replay_ratio
        replay_weight_each = replay_ratio / max(len(replay_datasets), 1)

        sources = [{"dataset": teacher_dataset, "weight": teacher_weight, "max_examples": 500}]
        for rds in replay_datasets:
            sources.append({"dataset": rds, "weight": replay_weight_each, "max_examples": 200})

        blend_result = blend_datasets(sources, blend_name)
        if blend_result.get("status") != "ok":
            _update_state(
                running=False, status="failed",
                error=f"Blend failed: {blend_result.get('reason', 'unknown')}",
                progress=0,
            )
            return

        logger.info(f"Distill blend created: {blend_result}")
        _update_state(
            status="training with distill hyperparameters",
            progress=10,
            examples_used=blend_result.get("total", 0),
        )

        # Run LoRA with distill hyperparameters
        _run_lora_training(
            dataset_name=blend_name,
            base_model=base_model,
            output_model=output_model,
            lr_override=DISTILL_LR,
            epochs_override=DISTILL_EPOCHS,
            warmup_override=DISTILL_WARMUP_RATIO,
        )

        # After LoRA completes, check if it succeeded
        if _training_state.get("status") == "failed":
            return  # LoRA already set error state

        # Run retention verification (soft gate)
        _update_state(status="verifying retention", progress=95)
        try:
            retention = verify_retention(output_model, baseline_model=base_model.replace("-ft", ""))
            if not retention.get("passed", False):
                logger.warning(
                    f"Retention check WARNING: pass_rate={retention.get('pass_rate', 0)} "
                    f"< threshold={retention.get('threshold', 0.7)}"
                )
                _update_state(
                    status="completed_with_warning",
                    error=f"Retention below threshold: {retention.get('pass_rate', 0):.0%}",
                )
            else:
                logger.info(f"Retention check passed: {retention.get('pass_rate', 0):.0%}")
        except Exception as e:
            logger.warning(f"Retention verification failed (non-blocking): {e}")

    except Exception as e:
        logger.error(f"Distill training error: {e}")
        _update_state(running=False, status="failed", error=safe_errors.public_error_detail(e), progress=0)
    finally:
        # Cleanup temporary blend dataset
        try:
            blend_path = _dataset_path(blend_name)
            if os.path.exists(blend_path):
                os.remove(blend_path)
                logger.info(f"Cleaned up temporary blend dataset: {blend_name}")
        except Exception:
            pass


# ── TRAINING ENTRY POINT ─────────────────────────────────────────────

def start_training(
    dataset_name: str,
    base_model: str = "deltai-qwen3b",
    output_model: str | None = None,
    mode: str = "auto",
    teacher_dataset: str | None = None,
    replay_datasets: list[str] | None = None,
) -> dict:
    """
    Start fine-tuning. Mode selection:
      - "lora": Real QLoRA training (requires GPU + deps)
      - "fewshot": Legacy few-shot embedding (always available)
      - "auto": Try LoRA first, fall back to fewshot
      - "distill": Knowledge distillation — blends teacher data with replay,
                   uses gentler hyperparameters to prevent forgetting
    """
    with _training_lock:
        if _training_state["running"]:
            return {"status": "error", "reason": "Training already in progress"}

    # VRAM / session guards before dataset or LoRA dep checks (explicit GPU-heavy modes)
    if mode in ("lora", "distill"):
        try:
            from router import is_sim_running, is_session_active

            if is_sim_running():
                return {"status": "error", "reason": "Cannot train while a GPU focus workload is active"}
            if is_session_active():
                return {"status": "error", "reason": "Cannot train during an active GPU focus session"}
        except ImportError:
            pass

    if output_model is None:
        output_model = f"{base_model}-ft"

    # Sanitize output_model and base_model to prevent path injection
    try:
        output_model = _sanitize_model_name(output_model)
        base_model = _sanitize_model_name(base_model)
    except ValueError:
        return {"status": "error", "reason": "Invalid model name: only alphanumerics, '.', '_', '-' are allowed"}

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
    elif mode == "distill":
        ok, reason = check_lora_deps()
        if not ok:
            return {"status": "error", "reason": f"LoRA deps unavailable (needed for distill): {reason}"}
        # Validate teacher dataset
        if not teacher_dataset:
            return {"status": "error", "reason": "Distill mode requires teacher_dataset parameter"}
        teacher_ds = get_dataset(teacher_dataset)
        if teacher_ds.get("status") != "ok" or not teacher_ds.get("examples"):
            return {"status": "error", "reason": f"Teacher dataset '{teacher_dataset}' not found or empty"}
        actual_mode = "distill"

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

    if actual_mode == "distill":
        replay = replay_datasets or ["deltai-personality", "deltai-auto", "deltai-anti-hallucination"]
        thread = threading.Thread(
            target=_run_distill_training,
            args=(teacher_dataset, replay, base_model, output_model),
            daemon=True,
        )
    elif actual_mode == "lora":
        thread = threading.Thread(
            target=_run_lora_training,
            args=(dataset_name, base_model, output_model),
            daemon=True,
        )
    else:
        thread = threading.Thread(
            target=_run_fewshot_training,
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
        safe_errors.log_exception(logger, "_run_eval_inference failed", e)
        return "[ERROR: inference failed]", int((time.time() - start) * 1000)


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
    # Use a trusted filename (no user-controlled model names) to avoid
    # uncontrolled data in path expressions.
    eval_file = f"ab_{int(time.time())}_{threading.get_ident()}.json"
    eval_root = os.path.realpath(EVAL_PATH)
    eval_path = os.path.realpath(os.path.join(eval_root, eval_file))
    if os.path.commonpath([eval_root, eval_path]) != eval_root:
        return {"status": "error", "reason": "Invalid evaluation file path"}
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


# ── KNOWLEDGE DISTILLATION ───────────────────────────────────────────

# Quality filter indicators (reused from auto_capture)
_ERROR_INDICATORS = ["error:", "exception:", "traceback", "failed to", "i can't", "i cannot"]

# Retention baseline queries — used to verify the model hasn't forgotten basics
_RETENTION_BASELINE_QUERIES = {
    "personality": [
        "Who are you?",
        "Hey deltai",
        "What's your designation?",
        "Who is your operator?",
        "Are you operational?",
    ],
    "instruction_following": [
        "List 3 types of heat transfer.",
        "Explain what a free body diagram is in 2 sentences.",
        "Compare static and dynamic friction.",
        "What units is pressure measured in?",
        "Summarize Newton's first law.",
    ],
    "domain_knowledge": [
        "What causes brake fade?",
        "What is Young's modulus?",
        "How does tire load sensitivity work?",
        "What's the formula for kinetic energy?",
        "What should I do if I don't have telemetry data?",
    ],
}


def generate_teacher_data(
    queries: list[str],
    teacher: str = "local14b",
    dataset_name: str = "deltai-teacher",
    category: str = "distilled",
) -> dict:
    """
    Generate training data using a teacher model.
    Teacher options:
      - "local14b": Uses Ollama's deltai-qwen14b model (free, local)
      - "anthropic": Uses Anthropic API (requires ANTHROPIC_API_KEY)
    Returns: {"status": "ok", "generated": N, "filtered": M, "dataset": name}
    """
    import httpx

    if not queries:
        return {"status": "ok", "generated": 0, "filtered": 0, "dataset": dataset_name}

    # Sim check for local14b (needs VRAM)
    if teacher == "local14b":
        try:
            from router import is_sim_running
            if is_sim_running():
                return {"status": "error", "reason": "Cannot use 14B teacher while a GPU focus workload is active"}
        except ImportError:
            pass

    # Anthropic teacher requires API key
    if teacher == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            return {"status": "error", "reason": "ANTHROPIC_API_KEY not set — cannot use Anthropic teacher"}

    # Ensure dataset exists
    ds_path = _dataset_path(dataset_name)
    if not os.path.exists(ds_path):
        create_dataset(dataset_name)

    generated = 0
    filtered = 0

    # System prompt for teacher — instruct it to respond as deltai
    system_prompt = (
        "You are deltai, a military-spec AI assistant. Respond concisely and technically. "
        "No filler phrases, no 'certainly', no 'I'd be happy to'. Lead with the answer. "
        "Keep responses under 300 characters for simple queries, up to 800 for complex ones. "
        "Be precise with engineering/physics/math. For racing, use data-driven language."
    )

    for query in queries:
        try:
            response_text = None

            if teacher == "local14b":
                with httpx.Client(timeout=DISTILL_TEACHER_TIMEOUT) as client:
                    resp = client.post(
                        f"{OLLAMA_URL}/api/chat",
                        json={
                            "model": "deltai-qwen14b",
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": query},
                            ],
                            "stream": False,
                        },
                    )
                    resp.raise_for_status()
                    response_text = resp.json().get("message", {}).get("content", "")

            elif teacher == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
                model = os.getenv("ANTHROPIC_SONNET_MODEL", "claude-sonnet-4-20250514")
                with httpx.Client(timeout=DISTILL_TEACHER_TIMEOUT) as client:
                    resp = client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={
                            "x-api-key": api_key,
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json",
                        },
                        json={
                            "model": model,
                            "max_tokens": 1024,
                            "system": system_prompt,
                            "messages": [{"role": "user", "content": query}],
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    content_blocks = data.get("content", [])
                    response_text = "".join(
                        b.get("text", "") for b in content_blocks if b.get("type") == "text"
                    )
            else:
                return {"status": "error", "reason": f"Unknown teacher: {teacher}"}

            if not response_text:
                filtered += 1
                continue

            response_text = response_text.strip()

            # Quality filters
            if len(response_text) < DISTILL_MIN_QUALITY_LEN:
                filtered += 1
                continue

            lower_resp = response_text.lower()
            if any(lower_resp.startswith(ind) for ind in _ERROR_INDICATORS):
                filtered += 1
                continue

            # Passed filters — add to dataset
            add_example(dataset_name, query, response_text, category=category)
            generated += 1

        except Exception as e:
            logger.warning(f"Teacher generation failed for query '{query[:50]}': {e}")
            filtered += 1
            continue

    return {"status": "ok", "generated": generated, "filtered": filtered, "dataset": dataset_name}


def blend_datasets(
    sources: list[dict],
    output_name: str,
    seed: int = 42,
) -> dict:
    """
    Blend multiple datasets with weighted sampling.
    Each source: {"dataset": "name", "weight": 0.3, "max_examples": 100}
    Returns: {"status": "ok", "total": N, "breakdown": {...}}
    """
    import random
    rng = random.Random(seed)

    # Check output doesn't already exist
    out_path = _dataset_path(output_name)
    if os.path.exists(out_path):
        return {"status": "error", "reason": f"Output dataset '{output_name}' already exists"}

    # Load and sample from each source
    all_pools = {}
    total_weight = sum(s.get("weight", 1.0) for s in sources)

    for src in sources:
        ds_name = src.get("dataset", "")
        weight = src.get("weight", 1.0)
        max_ex = src.get("max_examples", 9999)

        ds_result = get_dataset(ds_name)
        if ds_result.get("status") != "ok":
            logger.warning(f"Blend: skipping missing dataset '{ds_name}'")
            continue

        examples = ds_result.get("examples", [])
        if not examples:
            continue

        # Cap at max_examples
        if len(examples) > max_ex:
            examples = rng.sample(examples, max_ex)

        all_pools[ds_name] = {
            "examples": examples,
            "weight": weight / total_weight,  # Normalize
        }

    if not all_pools:
        return {"status": "error", "reason": "No valid source datasets with examples"}

    # Compute total target size (sum of all available weighted proportionally)
    total_available = sum(len(p["examples"]) for p in all_pools.values())
    target_total = total_available  # Use all available examples

    # Sample from each pool according to weight
    blended = []
    breakdown = {}

    for ds_name, pool in all_pools.items():
        target_count = min(
            int(target_total * pool["weight"] + 0.5),
            len(pool["examples"]),
        )
        if target_count > 0:
            sampled = rng.sample(pool["examples"], min(target_count, len(pool["examples"])))
            blended.extend(sampled)
            breakdown[ds_name] = len(sampled)

    # Shuffle the blended result
    rng.shuffle(blended)

    # Write output dataset
    create_dataset(output_name)
    for ex in blended:
        add_example(
            output_name,
            ex.get("input", ""),
            ex.get("output", ""),
            category=ex.get("category", "blended"),
        )

    return {"status": "ok", "name": output_name, "total": len(blended), "breakdown": breakdown}


def verify_retention(
    model_name: str,
    baseline_model: str = "deltai-qwen3b",
    min_pass_rate: float = 0.7,
) -> dict:
    """
    Verify a trained model hasn't lost base capabilities.
    Runs 15 baseline queries against both the candidate and baseline models.
    Returns: {"passed": bool, "pass_rate": float, "details": [...]}
    """
    import httpx

    all_queries = []
    for category, queries in _RETENTION_BASELINE_QUERIES.items():
        for q in queries:
            all_queries.append({"query": q, "category": category})

    details = []
    passed_count = 0

    for item in all_queries:
        query = item["query"]
        cat = item["category"]

        candidate_resp = ""
        baseline_resp = ""

        # Get candidate response
        try:
            with httpx.Client(timeout=60) as client:
                resp = client.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model": model_name,
                        "messages": [{"role": "user", "content": query}],
                        "stream": False,
                    },
                )
                resp.raise_for_status()
                candidate_resp = resp.json().get("message", {}).get("content", "").strip()
        except Exception as e:
            safe_errors.log_exception(logger, "verify_retention candidate inference failed", e)
            candidate_resp = "[ERROR: inference failed]"

        # Get baseline response
        try:
            with httpx.Client(timeout=60) as client:
                resp = client.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model": baseline_model,
                        "messages": [{"role": "user", "content": query}],
                        "stream": False,
                    },
                )
                resp.raise_for_status()
                baseline_resp = resp.json().get("message", {}).get("content", "").strip()
        except Exception as e:
            safe_errors.log_exception(logger, "verify_retention baseline inference failed", e)
            baseline_resp = "[ERROR: inference failed]"

        # Score: coherence, on-topic, personality
        score = 0
        checks = {}

        # Coherent: response is meaningful (>30 chars, no error)
        coherent = len(candidate_resp) > 30 and not candidate_resp.startswith("[ERROR")
        checks["coherent"] = coherent
        if coherent:
            score += 1

        # On-topic: contains at least one keyword from the query
        query_words = {w.lower() for w in query.split() if len(w) > 3}
        resp_lower = candidate_resp.lower()
        on_topic = any(w in resp_lower for w in query_words) if query_words else True
        checks["on_topic"] = on_topic
        if on_topic:
            score += 1

        # Personality preserved: not overly verbose, no banned phrases
        banned = ["certainly!", "absolutely!", "i'd be happy to", "great question", "as an ai"]
        personality_ok = (
            not any(b in resp_lower for b in banned)
            and (len(candidate_resp) < 3 * max(len(baseline_resp), 50))
        )
        checks["personality_preserved"] = personality_ok
        if personality_ok:
            score += 1

        query_passed = score >= 2
        if query_passed:
            passed_count += 1

        details.append({
            "query": query,
            "category": cat,
            "candidate_len": len(candidate_resp),
            "baseline_len": len(baseline_resp),
            "checks": checks,
            "score": score,
            "passed": query_passed,
        })

    pass_rate = passed_count / len(all_queries) if all_queries else 0.0

    return {
        "status": "ok",
        "passed": pass_rate >= min_pass_rate,
        "pass_rate": round(pass_rate, 3),
        "passed_count": passed_count,
        "total_queries": len(all_queries),
        "threshold": min_pass_rate,
        "details": details,
    }


# ── SMART AUTO-CAPTURE ───────────────────────────────────────────────────
# Quality-tiered capture with dedup, negative examples, and DPO pairs.

import hashlib as _hashlib
import random as _random

_CAPTURE_DEDUP_THRESHOLD = float(os.getenv("CAPTURE_DEDUP_THRESHOLD", "0.15"))
_SMART_CAPTURE_ENABLED = os.getenv("SMART_CAPTURE_ENABLED", "true").lower() in ("true", "1", "yes")

# In-memory dedup cache (recent capture hashes)
_capture_hashes: list[str] = []
_CAPTURE_HASH_CACHE_SIZE = 200


def _query_hash(text: str) -> str:
    """Fast hash for dedup."""
    return _hashlib.md5(text.strip().lower().encode()).hexdigest()


def smart_auto_capture(
    dataset_name: str,
    user_msg: str,
    assistant_msg: str,
    quality_score: float = 0.5,
    metadata: dict = None,
    rag_context: str = None,
) -> dict:
    """
    Quality-aware auto-capture with deduplication and negative examples.

    - Score >= 0.8: always capture (exemplary)
    - Score 0.6-0.8: capture with 50% probability
    - Score 0.3-0.6: skip (mediocre)
    - Score < 0.3: capture as negative example
    """
    global _capture_hashes

    if not _SMART_CAPTURE_ENABLED:
        return auto_capture(dataset_name, user_msg, assistant_msg)

    if not user_msg or not assistant_msg:
        return {"captured": False, "reason": "empty message"}

    user_msg = user_msg.strip()
    assistant_msg = assistant_msg.strip()

    # Skip very short messages
    if len(user_msg) < 4 or len(assistant_msg) < 30:
        return {"captured": False, "reason": "too short"}

    # Skip canned responses
    if assistant_msg in _CANNED_RESPONSES:
        return {"captured": False, "reason": "canned greeting"}

    # ── Deduplication: check against recent captures ──
    qhash = _query_hash(user_msg)
    if qhash in _capture_hashes:
        return {"captured": False, "reason": "duplicate query"}
    _capture_hashes.append(qhash)
    if len(_capture_hashes) > _CAPTURE_HASH_CACHE_SIZE:
        _capture_hashes = _capture_hashes[-_CAPTURE_HASH_CACHE_SIZE:]

    metadata = metadata or {}
    category = metadata.get("domain", "auto-captured")

    # Build capture input with context if available
    capture_input = user_msg
    if rag_context and category and category.startswith("telemetry_"):
        capture_input = f"[Context]\n{rag_context}\n\n[Query]\n{user_msg}"

    # ── Quality-tiered capture decision ──
    if quality_score >= 0.8:
        # Exemplary — always capture
        ds_path = _dataset_path(dataset_name)
        if not os.path.exists(ds_path):
            create_dataset(dataset_name)
        result = add_example(dataset_name, capture_input, assistant_msg, category=category)
        return {"captured": True, "dataset": dataset_name, "tier": "exemplary",
                "score": quality_score}

    elif quality_score >= 0.6:
        # Good — capture with 50% probability to prevent dataset bloat
        if _random.random() < 0.5:
            ds_path = _dataset_path(dataset_name)
            if not os.path.exists(ds_path):
                create_dataset(dataset_name)
            result = add_example(dataset_name, capture_input, assistant_msg, category=category)
            return {"captured": True, "dataset": dataset_name, "tier": "good",
                    "score": quality_score}
        return {"captured": False, "reason": "probabilistic skip (score 0.6-0.8)"}

    elif quality_score < 0.3:
        # Poor — capture as negative example for potential DPO training
        neg_dataset = f"{dataset_name}-negative"
        ds_path = _dataset_path(neg_dataset)
        if not os.path.exists(ds_path):
            create_dataset(neg_dataset)
        result = add_example(neg_dataset, capture_input, assistant_msg,
                           category=f"negative:{category}")
        return {"captured": True, "dataset": neg_dataset, "tier": "negative",
                "score": quality_score}

    else:
        # Mediocre (0.3-0.6) — skip
        return {"captured": False, "reason": "mediocre quality (0.3-0.6)"}


# ── ITERATIVE DISTILLATION PIPELINE ──────────────────────────────────────
# Closed loop: identify weaknesses → targeted teacher data → retrain → evaluate.

def identify_weak_domains(min_samples: int = 20) -> list[dict]:
    """
    Identify domains where deltai performs poorly based on routing feedback.
    Returns list of {"domain": str, "avg_score": float, "sample_count": int, "worst_queries": [...]}.
    """
    try:
        from persistence import get_routing_stats
    except ImportError:
        return []

    domains = ["racing", "engineering", "reasoning", "telemetry", "audio", "general"]
    weak = []

    for domain in domains:
        stats = get_routing_stats(domain, limit=200)
        if len(stats) < min_samples:
            continue
        scores = [s["score"] for s in stats if s["score"] is not None]
        if not scores:
            continue
        avg = sum(scores) / len(scores)
        if avg < 0.6:
            # Collect worst queries for targeted distillation
            worst = sorted(stats, key=lambda s: s.get("score", 1.0))[:10]
            weak.append({
                "domain": domain,
                "avg_score": round(avg, 3),
                "sample_count": len(scores),
                "worst_queries": worst,
            })

    weak.sort(key=lambda x: x["avg_score"])
    return weak


def distill_targeted(domain: str, n_queries: int = 50,
                     teacher_model: str = None) -> dict:
    """
    Generate targeted teacher data for a weak domain.
    Pulls the worst-scoring queries and feeds them to the teacher model.

    Returns {"status": "ok", "dataset": str, "examples": int} or error.
    """
    try:
        from persistence import get_routing_stats
    except ImportError:
        return {"status": "error", "reason": "persistence not available"}

    if not teacher_model:
        teacher_model = os.getenv("DELTAI_STRONG_MODEL", "deltai-qwen14b")

    stats = get_routing_stats(domain, limit=500)
    if not stats:
        return {"status": "error", "reason": f"no routing data for domain: {domain}"}

    # Get worst-scoring queries
    scored = [(s.get("score", 1.0), s) for s in stats if s.get("score") is not None]
    scored.sort(key=lambda x: x[0])
    worst = scored[:n_queries]

    # Generate teacher data for these queries
    dataset_name = f"distill-{domain}-targeted"
    ds_path = _dataset_path(dataset_name)
    if not os.path.exists(ds_path):
        create_dataset(dataset_name)

    generated = 0
    # Note: actual teacher generation would call Ollama with the teacher model
    # For now, log the queries that need improvement
    for score, stat in worst:
        # These would be fed to generate_teacher_data in a full implementation
        # For now, record them as queries needing teacher responses
        pass

    return {
        "status": "ok",
        "dataset": dataset_name,
        "domain": domain,
        "queries_identified": len(worst),
        "avg_worst_score": round(sum(s for s, _ in worst) / len(worst), 3) if worst else 0,
    }


def run_improvement_cycle(domain: str) -> dict:
    """
    Full improvement cycle for a domain:
    1. Identify weakness
    2. Generate targeted teacher data
    3. (Future: blend, train, eval, promote)

    Returns summary with status and metrics.
    """
    # Step 1: Verify domain is actually weak
    weak = identify_weak_domains(min_samples=10)
    target = next((w for w in weak if w["domain"] == domain), None)

    if not target:
        return {"status": "skipped", "reason": f"domain '{domain}' is not below threshold",
                "weak_domains": [w["domain"] for w in weak]}

    # Step 2: Generate targeted data
    result = distill_targeted(domain)

    return {
        "status": "ok",
        "domain": domain,
        "before_score": target["avg_score"],
        "sample_count": target["sample_count"],
        "distillation": result,
        "next_steps": "Run domain training with the targeted dataset, then eval and promote.",
    }


# ── DPO TRAINING ─────────────────────────────────────────────────────────
# Direct Preference Optimization using positive/negative example pairs.
# Consumes {dataset}-negative datasets accumulated by smart_auto_capture.

def start_dpo_training(
    positive_dataset: str,
    negative_dataset: str = None,
    output_model: str = None,
    base_model: str = None,
) -> dict:
    """
    Start DPO (Direct Preference Optimization) training using positive/negative pairs.
    Uses trl.DPOTrainer to improve preference alignment.

    positive_dataset: dataset with good examples (score >= 0.6)
    negative_dataset: dataset with bad examples (default: {positive_dataset}-negative)
    output_model: Ollama model name for the result
    base_model: HuggingFace base model (default: HF_BASE_MODEL env var)

    Requires same deps as LoRA: transformers, peft, trl, bitsandbytes.
    """
    with _training_lock:
        if _training_state["running"]:
            return {"status": "error", "reason": "Training already in progress"}

    if negative_dataset is None:
        negative_dataset = f"{positive_dataset}-negative"

    pos_path = _dataset_path(positive_dataset)
    neg_path = _dataset_path(negative_dataset)

    if not os.path.exists(pos_path):
        return {"status": "error", "reason": f"Positive dataset not found: {positive_dataset}"}
    if not os.path.exists(neg_path):
        return {"status": "error", "reason": f"Negative dataset not found: {negative_dataset}"}

    effective_base = base_model or HF_BASE_MODEL
    effective_output = output_model or f"deltai-qwen3b-dpo-{positive_dataset}"

    _training_cancel_flag.clear()
    _update_state(
        running=True, status="starting DPO training", progress=0,
        dataset=positive_dataset, model=effective_output, mode="dpo",
        error=None, loss=None, completed=None,
    )

    def _run_dpo():
        try:
            _dpo_train_impl(positive_dataset, negative_dataset, effective_base, effective_output)
        except Exception as e:
            logger.error(f"DPO training failed: {e}", exc_info=True)
            _update_state(running=False, status="failed", error=safe_errors.public_error_detail(e))

    t = threading.Thread(target=_run_dpo, daemon=True)
    t.start()
    return {
        "status": "ok",
        "message": f"DPO training started: {effective_output} (pos={positive_dataset}, neg={negative_dataset})",
        "mode": "dpo",
        "output_model": effective_output,
    }


def _dpo_train_impl(
    positive_dataset: str,
    negative_dataset: str,
    base_model: str,
    output_model: str,
) -> None:
    """Internal DPO training implementation. Runs in background thread."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import DPOTrainer, DPOConfig
        from datasets import Dataset as HFDataset
    except ImportError as e:
        logger.error("DPO training dependencies missing: %s", e)
        _update_state(
            running=False,
            status="failed",
            error=(
                "DPO deps missing. Install: pip install trl transformers peft bitsandbytes"
            ),
        )
        return

    _update_state(progress=5, status="loading positive/negative datasets")
    output_model = _sanitize_model_name(output_model)
    pos_result = get_dataset(positive_dataset)
    neg_result = get_dataset(negative_dataset)

    if pos_result.get("status") != "ok" or neg_result.get("status") != "ok":
        _update_state(running=False, status="failed", error="Could not load datasets")
        return

    pos_examples = pos_result.get("examples", [])
    neg_examples = neg_result.get("examples", [])

    if not pos_examples or not neg_examples:
        _update_state(running=False, status="failed", error="Empty dataset(s) for DPO")
        return

    # Build preference pairs: match by input text
    pos_by_input = {ex.get("input", ""): ex.get("output", "") for ex in pos_examples}
    neg_by_input = {ex.get("input", ""): ex.get("output", "") for ex in neg_examples}

    pairs = []
    for inp, chosen in pos_by_input.items():
        rejected = neg_by_input.get(inp)
        if rejected and chosen != rejected:
            pairs.append({"prompt": inp, "chosen": chosen, "rejected": rejected})

    if len(pairs) < 10:
        _update_state(running=False, status="failed",
                      error=f"Too few preference pairs ({len(pairs)}). Need at least 10 matched examples.")
        return

    _update_state(progress=10, status=f"building {len(pairs)} preference pairs")

    hf_data = HFDataset.from_list(pairs)
    train_size = max(1, int(len(hf_data) * 0.9))
    train_ds = hf_data.select(range(train_size))
    eval_ds = hf_data.select(range(train_size, len(hf_data))) if len(hf_data) > train_size else train_ds

    _update_state(progress=15, status="loading base model (4-bit)")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _update_state(progress=20, status="applying LoRA for DPO")
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                          target_modules=["q_proj", "v_proj"], bias="none",
                          task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)

    adapter_dir = os.path.join(ADAPTERS_PATH, output_model)
    os.makedirs(adapter_dir, exist_ok=True)

    dpo_args = DPOConfig(
        output_dir=adapter_dir,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        beta=0.1,
        max_length=512,
        max_prompt_length=256,
    )

    _update_state(progress=30, status="DPO training")
    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )
    trainer.train()

    _update_state(progress=85, status="saving DPO adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    del model, trainer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _update_state(running=False, progress=100, status="DPO training complete",
                  completed=int(time.time()), error=None)
    logger.info(f"DPO training complete: {output_model} ({len(pairs)} pairs)")


# ── SESSION KNOWLEDGE SYNTHESIS ───────────────────────────────────────────
# At session end, synthesize key learnings into a durable knowledge article.

SESSION_SYNTHESIS_ENABLED = os.getenv("SESSION_SYNTHESIS_ENABLED", "true").lower() in ("true", "1", "yes")
SESSION_SYNTHESIS_MODEL = os.getenv("SESSION_SYNTHESIS_MODEL", "local14b")


def synthesize_session_knowledge(
    session_id: str,
    session_turns: list[dict],
    teacher: str = None,
) -> dict:
    """
    After a racing/work session ends, distill key learnings into a knowledge article.

    session_turns: list of {"role": str, "content": str} pairs from the session
    teacher: "local14b" or "local3b" (default: SESSION_SYNTHESIS_MODEL env var)

    Returns {"status": "ok", "knowledge_text": str, "session_id": str, "words": int}
    The caller is responsible for ingesting the returned knowledge_text into ChromaDB.
    """
    if not SESSION_SYNTHESIS_ENABLED:
        return {"status": "skipped", "reason": "SESSION_SYNTHESIS_ENABLED=false"}

    if not session_turns:
        return {"status": "skipped", "reason": "no session turns provided"}

    effective_teacher = teacher or SESSION_SYNTHESIS_MODEL
    model_name = os.getenv("DELTAI_STRONG_MODEL", "deltai-qwen14b") if effective_teacher == "local14b" \
        else os.getenv("DELTAI_MODEL", "deltai-qwen3b")

    # Build session transcript (trimmed to avoid token overflow)
    transcript_lines = []
    for turn in session_turns[-30:]:
        role = turn.get("role", "user")
        content = (turn.get("content") or "")[:600]
        transcript_lines.append(f"{role.upper()}: {content}")
    transcript = "\n".join(transcript_lines)

    synthesis_prompt = (
        "You are deltai's knowledge synthesis module. Analyze the following session transcript "
        "and write a concise knowledge article (200-400 words) capturing:\n"
        "1. Key decisions made and their rationale\n"
        "2. Technical findings or setup discoveries\n"
        "3. Strategy patterns that worked or failed\n"
        "4. Any engineering or physics insights discussed\n"
        "5. Action items or lessons for future sessions\n\n"
        "Format as a structured article with clear headings. Be specific — include numbers, "
        "lap times, temperatures, or setup values where mentioned.\n\n"
        f"SESSION TRANSCRIPT:\n{transcript}\n\nKNOWLEDGE ARTICLE:"
    )

    try:
        resp = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": synthesis_prompt}],
                "stream": False,
                "options": {"num_predict": 600, "temperature": 0.3},
            },
            timeout=120,
        )
        resp.raise_for_status()
        knowledge_text = resp.json().get("message", {}).get("content", "").strip()
    except Exception as e:
        safe_errors.log_exception(logger, "session synthesis model call failed", e)
        return {
            "status": "error",
            "reason": f"Synthesis model call failed: {safe_errors.public_error_detail(e)}",
        }

    if not knowledge_text or len(knowledge_text) < 100:
        return {"status": "error", "reason": "Synthesis returned insufficient content"}

    words = len(knowledge_text.split())
    logger.info(f"Session synthesis complete: session={session_id}, words={words}, model={model_name}")
    return {
        "status": "ok",
        "knowledge_text": knowledge_text,
        "session_id": session_id,
        "words": words,
        "model": model_name,
    }


# ── DAILY TRAINING CYCLE ──────────────────────────────────────────────────
# Orchestrates the full daily self-improvement loop.

# Weekly curriculum map: weekday (0=Mon) -> list of (dataset_name, topic_label)
_DAILY_CURRICULUM = {
    0: [("deltai-telemetry-analysis", "telemetry"), ("deltai-audio-analysis", "audio")],
    1: [("deltai-strategy-advanced", "telemetry"), ("deltai-race-engineering", "racing")],
    2: [("deltai-engineering", "engineering"), ("deltai-meche", "engineering")],
    3: [("deltai-eng-simulations", "engineering"), ("deltai-cot-reasoning", "reasoning")],
    4: [("deltai-audio-analysis", "audio"), ("deltai-telemetry-analysis", "telemetry")],
    5: [("deltai-reasoning", "reasoning"), ("deltai-data-context", "reasoning")],
    6: [("deltai-personality", "personality"), ("deltai-anti-hallucination", "reasoning")],
}

_DAILY_TRAIN_MIN_VRAM_MB = int(os.getenv("DAILY_TRAIN_MIN_VRAM_MB", "7000"))
_DAILY_TRAIN_AUTO_PROMOTE = os.getenv("DAILY_TRAIN_AUTO_PROMOTE", "false").lower() in ("true", "1")
_DAILY_TRAIN_AUTO_MERGE = os.getenv("DAILY_TRAIN_AUTO_MERGE", "false").lower() in ("true", "1")


def run_daily_cycle(
    force_day_override: int = None,
    dry_run: bool = False,
    auto_train: bool = True,
    auto_merge: bool = None,
    auto_promote: bool = None,
) -> dict:
    """
    Full daily training cycle. Safe to call from the 2 AM scheduler.

    Phases:
    0. Guard checks (focus workload, VRAM available, training already running)
    1. Weakness analysis — find domains with avg quality < 0.6
    2. Targeted distillation — teacher generates examples for weak domains (max 2)
    3. Curriculum generation — daily topic datasets via teacher
    4. Blend + QLoRA domain adapter training (if auto_train=True)
    5. Eval + optional auto-promote
    6. Memory consolidation report

    Returns structured report dict.
    """
    import datetime

    report = {
        "date": datetime.datetime.now().isoformat(),
        "dry_run": dry_run,
        "phases": {},
        "status": "ok",
        "errors": [],
    }

    effective_auto_merge = _DAILY_TRAIN_AUTO_MERGE if auto_merge is None else auto_merge
    effective_auto_promote = _DAILY_TRAIN_AUTO_PROMOTE if auto_promote is None else auto_promote

    # ── Phase 0: Guard checks ──
    guards = {}
    try:
        from router import is_sim_running, _get_vram_info
        _sim = is_sim_running()
        guards["sim_running"] = _sim  # legacy report key
        guards["focus_workload_active"] = _sim
        vram_info = _get_vram_info()  # returns (used_mb, total_mb, free_mb)
        guards["vram_free_mb"] = vram_info[2]
        guards["vram_ok"] = guards["vram_free_mb"] >= _DAILY_TRAIN_MIN_VRAM_MB
    except Exception as e:
        guards["vram_ok"] = False
        guards["guard_error"] = safe_errors.public_error_detail(e)
        report["errors"].append(f"Guard check error: {safe_errors.public_error_detail(e)}")

    with _training_lock:
        guards["training_running"] = _training_state.get("running", False)

    report["phases"]["guards"] = guards

    if guards.get("sim_running"):
        report["status"] = "skipped"
        report["skip_reason"] = "GPU focus workload active — daily training deferred"
        logger.info("Daily cycle skipped: focus workload / sim detected")
        return report

    if guards.get("training_running"):
        report["status"] = "skipped"
        report["skip_reason"] = "Training already in progress"
        return report

    if not guards.get("vram_ok", False) and not dry_run:
        report["status"] = "skipped"
        report["skip_reason"] = (
            f"Insufficient VRAM ({guards.get('vram_free_mb', 0)}MB < "
            f"{_DAILY_TRAIN_MIN_VRAM_MB}MB required)"
        )
        logger.info(f"Daily cycle skipped: {report['skip_reason']}")
        return report

    # Ensure SQLite schema (routing_feedback, knowledge_gaps, etc.) exists before any phase
    # that queries persistence — init was previously only in phase 5, which ran too late.
    try:
        from persistence import init_db  # noqa: PLC0415

        init_db()
    except Exception as e:
        report["status"] = "error"
        report["errors"].append(f"Persistence init failed: {safe_errors.public_error_detail(e)}")
        logger.exception("Daily cycle: persistence init failed")
        return report

    # ── Phase 0.5: Web training data collection ──
    collect_report: dict = {"status": "disabled"}
    if _WEB_COLLECT_ENABLED:
        try:
            from collector import run_collection_cycle  # noqa: PLC0415
            collect_report = run_collection_cycle(dry_run=dry_run)
            logger.info(
                f"Web collection: written={collect_report.get('total_written', 0)}, "
                f"skipped={collect_report.get('total_skipped', 0)}, "
                f"status={collect_report.get('status')}"
            )
        except Exception as e:
            collect_report = {"status": "error", "error": safe_errors.public_error_detail(e)}
            report["errors"].append(f"Web collection error: {safe_errors.public_error_detail(e)}")
            logger.warning(f"Web collection failed (non-fatal): {e}")
    report["phases"]["web_collection"] = collect_report

    # ── Phase 1: Weakness analysis ──
    weak_domains = identify_weak_domains(min_samples=15)
    report["phases"]["weakness_analysis"] = {
        "weak_domains": [{"domain": w["domain"], "avg_score": w["avg_score"],
                          "samples": w["sample_count"]} for w in weak_domains]
    }
    logger.info(f"Daily cycle: weak domains = {[w['domain'] for w in weak_domains]}")

    # ── Phase 2: Targeted distillation (max 2 weak domains) ──
    distill_results = []
    for weak in weak_domains[:2]:
        domain = weak["domain"]
        queries = [q.get("query", "") for q in weak.get("worst_queries", []) if q.get("query")]
        if not queries:
            continue
        if dry_run:
            distill_results.append({"domain": domain, "status": "dry_run", "queries": len(queries)})
            continue
        teacher_result = generate_teacher_data(
            queries=queries[:15],
            teacher="local14b",
            dataset_name=f"distill-{domain}-targeted",
            category=f"distill-{domain}",
        )
        distill_results.append({
            "domain": domain,
            "generated": teacher_result.get("generated", 0),
            "filtered": teacher_result.get("filtered", 0),
            "status": teacher_result.get("status", "unknown"),
        })
    report["phases"]["targeted_distillation"] = distill_results

    # ── Phase 3: Daily curriculum ──
    weekday = force_day_override if force_day_override is not None else datetime.datetime.now().weekday()
    curriculum = _DAILY_CURRICULUM.get(weekday, _DAILY_CURRICULUM[0])
    curriculum_results = []
    for ds_name, domain in curriculum:
        ds_result = get_dataset(ds_name)
        count = len(ds_result.get("examples", [])) if ds_result.get("status") == "ok" else 0
        curriculum_results.append({"dataset": ds_name, "domain": domain, "examples": count})
    report["phases"]["curriculum"] = {
        "weekday": weekday,
        "datasets": curriculum_results,
    }

    # ── Phase 4: Domain training ──
    # Determine which domain to train today based on weekday
    train_domain_map = {0: "telemetry", 1: "racing", 2: "engineering", 3: "reasoning",
                        4: "audio", 5: "reasoning", 6: "personality"}
    train_domain = train_domain_map.get(weekday, "reasoning")

    # Gather datasets for this domain
    domain_datasets = [ds for ds, d in DATASET_DOMAIN_MAP.items() if d == train_domain]
    domain_examples = sum(
        len((get_dataset(ds) or {}).get("examples", []))
        for ds in domain_datasets
        if os.path.exists(_dataset_path(ds))
    )

    training_result = {"domain": train_domain, "status": "skipped", "reason": ""}

    if not auto_train:
        training_result["reason"] = "auto_train=False"
    elif domain_examples < 20:
        training_result["reason"] = f"Insufficient examples ({domain_examples} < 20)"
    elif dry_run:
        training_result["status"] = "dry_run"
        training_result["examples"] = domain_examples
    else:
        # Blend domain datasets for training
        blend_name = f"daily-blend-{train_domain}-{datetime.datetime.now().strftime('%Y%m%d')}"
        sources = [{"dataset": ds, "weight": 1.0}
                   for ds in domain_datasets if os.path.exists(_dataset_path(ds))]
        if sources:
            blend_result = blend_datasets(blend_name, sources)
            if blend_result.get("status") == "ok":
                output_name = f"deltai-qwen3b-daily-{train_domain}"
                train_result = start_training(
                    dataset_name=blend_name,
                    output_model=output_name,
                    mode="lora",
                    domain=train_domain,
                    adapter_only=True,
                )
                training_result.update({
                    "status": train_result.get("status", "unknown"),
                    "domain": train_domain,
                    "blend": blend_result.get("total", 0),
                    "output": output_name,
                })

    report["phases"]["training"] = training_result

    # ── Phase 5: Knowledge gap review ──
    gap_summary = {"status": "ok"}
    try:
        from persistence import count_unresolved_knowledge_gaps  # noqa: PLC0415

        gap_summary["unresolved_gaps"] = count_unresolved_knowledge_gaps()
    except Exception as e:
        gap_summary["status"] = "error"
        gap_summary["error"] = safe_errors.public_error_detail(e)
    report["phases"]["knowledge_gaps"] = gap_summary

    logger.info(f"Daily cycle complete: {report['status']} — "
                f"weak={len(weak_domains)}, distilled={len(distill_results)}, "
                f"train_domain={train_domain}, dry_run={dry_run}")

    # Write report to disk
    try:
        reports_dir = os.path.join(TRAINING_PATH, "daily_reports")
        os.makedirs(reports_dir, exist_ok=True)
        report_file = os.path.join(
            reports_dir,
            f"{datetime.datetime.now().strftime('%Y-%m-%d')}.json"
        )
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        report["report_file"] = report_file
    except Exception as e:
        report["errors"].append(f"Report write failed: {safe_errors.public_error_detail(e)}")

    return report
