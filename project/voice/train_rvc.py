"""
RVC v2 training pipeline for deltai custom voice.

Workflow:
    1. prepare_dataset() — Clean, split, resample training audio
    2. train() — Run RVC training (unloads Ollama, uses full GPU)
    3. export_model() — Extract lightweight inference model

Training audio should be placed in ~/deltai/data\\voice\\training_audio\\
as mono WAV files (44100Hz preferred, any sample rate accepted).
Target: 15-40 minutes total, 60% deltai dialogue / 40% BT-7274 dialogue.
"""

import logging
import os
import shutil
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import safe_errors

from .voice_config import DEFAULT_CONFIG, VoiceConfig

logger = logging.getLogger("deltai.voice.train")

# ── Training State ─────────────────────────────────────────────────────

@dataclass
class RVCTrainingState:
    """Tracks RVC training progress (accessible via API)."""
    status: str = "idle"           # idle, preparing, training, exporting, complete, error, aborted
    progress: float = 0.0          # 0.0 to 1.0
    current_epoch: int = 0
    total_epochs: int = 200
    loss_g: float = 0.0            # Generator loss
    loss_d: float = 0.0            # Discriminator loss
    error: str | None = None
    started_at: float | None = None
    elapsed_sec: float = 0.0
    dataset_stats: dict = field(default_factory=dict)

_training_state = RVCTrainingState()
_training_thread: threading.Thread | None = None
_training_abort = threading.Event()


def get_training_state() -> dict:
    """Return current training state as a dict (for API response)."""
    if _training_state.started_at and _training_state.status == "training":
        _training_state.elapsed_sec = time.time() - _training_state.started_at
    return {
        "status": _training_state.status,
        "progress": round(_training_state.progress, 3),
        "current_epoch": _training_state.current_epoch,
        "total_epochs": _training_state.total_epochs,
        "loss_g": round(_training_state.loss_g, 4),
        "loss_d": round(_training_state.loss_d, 4),
        "error": _training_state.error,
        "elapsed_sec": round(_training_state.elapsed_sec, 1),
        "dataset_stats": _training_state.dataset_stats,
    }


def is_training() -> bool:
    """Check if RVC training is currently running."""
    return _training_state.status in ("preparing", "training", "exporting")


# ── Dataset Preparation ────────────────────────────────────────────────

def prepare_dataset(
    audio_dir: str = None,
    output_dir: str = None,
    config: VoiceConfig | None = None,
) -> dict:
    """Prepare training audio: split, resample, normalize.

    Scans audio_dir for WAV files, processes them into training-ready segments.

    Args:
        audio_dir: Directory containing raw WAV files.
                   Default: ~/deltai/data\\voice\\training_audio\\
        output_dir: Where to save processed segments.
                    Default: ~/deltai/data\\voice\\training_processed\\
        config: Voice configuration.

    Returns:
        Stats dict: {n_files, n_segments, total_duration_sec, output_dir}
    """
    global _training_state
    config = config or DEFAULT_CONFIG

    audio_dir = Path(audio_dir or config.rvc.training_audio_dir)
    output_dir = Path(output_dir or str(audio_dir).replace("training_audio", "training_processed"))

    if not audio_dir.exists():
        raise FileNotFoundError(f"Training audio directory not found: {audio_dir}")

    _training_state.status = "preparing"
    _training_state.progress = 0.0

    # Scan for WAV files
    wav_files = sorted(audio_dir.glob("*.wav"))
    if not wav_files:
        _training_state.status = "idle"
        raise ValueError(f"No WAV files found in {audio_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    target_sr = config.rvc.training_sample_rate  # 40000 Hz for RVC v2
    min_dur = config.rvc.min_segment_duration     # 3 seconds
    max_dur = config.rvc.max_segment_duration     # 15 seconds

    n_segments = 0
    total_duration = 0.0
    processed_files = 0

    for i, wav_path in enumerate(wav_files):
        _training_state.progress = i / len(wav_files) * 0.9

        try:
            # Read WAV
            with wave.open(str(wav_path), "rb") as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)

            # Convert to float32
            if sample_width == 2:
                pcm = np.frombuffer(raw, dtype=np.int16)
                audio = pcm.astype(np.float32) / 32768.0
            elif sample_width == 4:
                pcm = np.frombuffer(raw, dtype=np.int32)
                audio = pcm.astype(np.float32) / 2147483648.0
            else:
                continue  # Skip unsupported formats

            # Convert to mono if stereo
            if n_channels == 2:
                audio = audio.reshape(-1, 2).mean(axis=1)

            # Resample to target SR
            if framerate != target_sr:
                from scipy.signal import resample
                n_target = int(len(audio) * target_sr / framerate)
                audio = resample(audio, n_target).astype(np.float32)

            # Normalize to -1dB peak
            peak = np.abs(audio).max()
            if peak > 0:
                target_peak = 10 ** (-1 / 20)  # -1dB
                audio = audio * (target_peak / peak)

            # Split into segments
            samples_per_seg = int(max_dur * target_sr)
            min_samples = int(min_dur * target_sr)
            offset = 0

            while offset < len(audio):
                segment = audio[offset:offset + samples_per_seg]
                if len(segment) < min_samples:
                    break

                seg_name = f"{wav_path.stem}_{n_segments:04d}.wav"
                seg_path = output_dir / seg_name

                seg_16bit = (segment * 32767).clip(-32768, 32767).astype(np.int16)
                with wave.open(str(seg_path), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(target_sr)
                    wf.writeframes(seg_16bit.tobytes())

                n_segments += 1
                total_duration += len(segment) / target_sr
                offset += samples_per_seg

            processed_files += 1

        except Exception as e:
            logger.warning("Failed to process %s: %s", wav_path.name, e)

    stats = {
        "n_files": processed_files,
        "n_files_total": len(wav_files),
        "n_segments": n_segments,
        "total_duration_sec": round(total_duration, 1),
        "target_sample_rate": target_sr,
        "output_dir": str(output_dir),
    }

    _training_state.dataset_stats = stats
    _training_state.progress = 1.0
    _training_state.status = "idle"

    logger.info(
        "Dataset prepared: %d files -> %d segments (%.1f min)",
        processed_files, n_segments, total_duration / 60,
    )
    return stats


# ── Training ───────────────────────────────────────────────────────────

def _check_sim_running() -> bool:
    """Check if a racing sim is running (blocks training)."""
    try:
        from router import is_sim_running
        return is_sim_running()
    except ImportError:
        return False


def _unload_ollama_models() -> None:
    """Unload all Ollama models from VRAM to free space for training."""
    try:
        import httpx
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        r = httpx.get(f"{ollama_url}/api/ps", timeout=10)
        if r.status_code == 200:
            data = r.json()
            models = data.get("models", [])
            for m in models:
                name = m.get("name", "")
                if name:
                    httpx.post(
                        f"{ollama_url}/api/generate",
                        json={"model": name, "keep_alive": 0},
                        timeout=15,
                    )
                    logger.info("Unloaded Ollama model: %s", name)
            time.sleep(3)
    except Exception as e:
        logger.warning("Failed to unload Ollama models: %s", e)


def train(
    dataset_dir: str = None,
    output_name: str = None,
    config: VoiceConfig | None = None,
) -> None:
    """Start RVC training in a background thread.

    Training requires the full GPU -- Ollama models are unloaded first.
    Blocks if sim is running.

    Args:
        dataset_dir: Directory with prepared training segments.
        output_name: Name for the trained model.
        config: Voice configuration.
    """
    global _training_thread, _training_state

    if is_training():
        raise RuntimeError("Training already in progress")

    if _check_sim_running():
        raise RuntimeError("Cannot train while racing sim is running")

    config = config or DEFAULT_CONFIG
    dataset_dir = dataset_dir or str(
        Path(config.rvc.training_audio_dir).parent / "training_processed"
    )
    output_name = output_name or config.rvc.model_name

    _training_abort.clear()
    _training_state = RVCTrainingState(
        status="training",
        total_epochs=config.rvc.training_epochs,
        started_at=time.time(),
    )

    def _train_worker():
        try:
            _do_train(dataset_dir, output_name, config)
        except Exception as e:
            _training_state.status = "error"
            _training_state.error = safe_errors.public_error_detail(e)
            logger.error("RVC training failed: %s", e)

    _training_thread = threading.Thread(target=_train_worker, daemon=True)
    _training_thread.start()


def _do_train(dataset_dir: str, output_name: str, config: VoiceConfig) -> None:
    """Actual training implementation (runs in background thread).

    Note: Full RVC v2 training requires the SynthesizerTrnMs model architecture
    and training scripts from the RVC project. The rvc-python library does not
    yet support Python 3.14. Until it does, training should be done using the
    standalone RVC WebUI, and the exported model placed in the models directory.
    """
    global _training_state

    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    segments = sorted(dataset_path.glob("*.wav"))
    if not segments:
        raise ValueError(f"No WAV segments found in {dataset_dir}")

    logger.info(
        "RVC training: %d segments, %d epochs, model=%s",
        len(segments), config.rvc.training_epochs, output_name,
    )

    _unload_ollama_models()

    # Check for RVC training availability
    _training_state.status = "error"
    _training_state.error = (
        "Full RVC v2 training requires rvc-python or vendored RVC training scripts. "
        "rvc-python is not yet compatible with Python 3.14. "
        "Train using the RVC WebUI (https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) "
        "and place the exported .pth + .index files in "
        f"~/deltai/data\\voice\\models\\{output_name}\\"
    )
    logger.warning(_training_state.error)


def stop_training() -> None:
    """Signal training to abort."""
    if is_training():
        _training_abort.set()
        _training_state.status = "aborted"
        logger.info("RVC training abort requested")


# ── Model Export ───────────────────────────────────────────────────────

def export_model(
    checkpoint_dir: str = None,
    output_name: str = None,
    config: VoiceConfig | None = None,
) -> dict:
    """Export a trained RVC checkpoint to a lightweight inference model.

    Strips optimizer states and discriminator weights to reduce file size.

    Args:
        checkpoint_dir: Directory containing training checkpoints.
        output_name: Name for the exported model.
        config: Voice configuration.

    Returns:
        Dict with export details: {model_path, index_path, size_mb}
    """
    config = config or DEFAULT_CONFIG
    output_name = output_name or config.rvc.model_name

    output_dir = Path(config.rvc.model_dir) / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(checkpoint_dir or str(
        Path(config.rvc.training_audio_dir).parent / "training_checkpoints"
    ))

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    checkpoints = sorted(checkpoint_dir.glob("G_*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No generator checkpoints found in {checkpoint_dir}")

    latest = checkpoints[-1]

    try:
        import torch

        ckpt = torch.load(str(latest), map_location="cpu", weights_only=False)

        export = {
            "weight": {},
            "config": ckpt.get("config", []),
            "info": f"deltai voice model exported from {latest.name}",
        }

        model_state = ckpt.get("model", ckpt.get("weight", {}))
        for k, v in model_state.items():
            if not k.startswith("optimizer"):
                export["weight"][k] = v

        model_path = output_dir / f"{output_name}.pth"
        torch.save(export, str(model_path))

        index_files = list(checkpoint_dir.glob("*.index"))
        index_path = None
        if index_files:
            index_path = output_dir / f"{output_name}.index"
            shutil.copy2(str(index_files[0]), str(index_path))

        size_mb = model_path.stat().st_size / 1e6

        result = {
            "model_path": str(model_path),
            "index_path": str(index_path) if index_path else None,
            "size_mb": round(size_mb, 1),
            "source_checkpoint": latest.name,
        }

        logger.info("Exported RVC model: %s (%.1f MB)", model_path.name, size_mb)
        return result

    except ImportError:
        raise RuntimeError("PyTorch required for model export")
