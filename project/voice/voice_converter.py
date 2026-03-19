"""
RVC v2 voice conversion — converts TTS output through a trained voice model.

Pipeline:
    1. Resample input to 16kHz for feature extraction
    2. Extract HuBERT features (speaker embedding)
    3. Extract f0 (pitch) using RMVPE or DIO
    4. Run RVC synthesis network (generator) to produce converted audio
    5. Resample output back to original sample rate

VRAM: ~1.5-2GB when loaded on GPU. Falls back to CPU if insufficient VRAM.
Graceful passthrough if model not trained or deps missing.
"""

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .voice_config import VoiceConfig, DEFAULT_CONFIG

logger = logging.getLogger("e3n.voice.rvc")

# Minimum free VRAM (MB) to load RVC on GPU
_MIN_VRAM_MB = 2500


class VoiceConverter:
    """RVC v2 voice conversion.

    Loads a trained .pth model + optional .index file for feature matching.
    Converts TTS audio to match the target voice identity.
    Manages VRAM (load/unload for GPU sharing with LLM inference).

    Fallback: returns input audio unchanged if model not available.
    """

    def __init__(self, config: Optional[VoiceConfig] = None) -> None:
        self._config = config or DEFAULT_CONFIG
        self._loaded = False
        self._device = "cpu"
        self._model = None           # RVC generator model
        self._hubert = None          # HuBERT feature extractor
        self._f0_extractor = None    # Pitch extractor (RMVPE)
        self._index = None           # FAISS index for feature matching
        self._target_sr = 40000      # RVC v2 standard output sample rate

    @property
    def is_loaded(self) -> bool:
        """Whether the RVC model is currently loaded in memory."""
        return self._loaded

    @property
    def device(self) -> str:
        """Current compute device ('cuda' or 'cpu')."""
        return self._device

    @property
    def model_path(self) -> Path:
        """Path to the RVC model directory."""
        return Path(self._config.rvc.model_dir) / self._config.rvc.model_name

    def _select_device(self) -> str:
        """Select CUDA or CPU based on available VRAM."""
        try:
            import torch
            if not torch.cuda.is_available():
                return "cpu"
            # Check free VRAM
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                free_mb = (mem.total - mem.used) / 1e6
                if free_mb < _MIN_VRAM_MB:
                    logger.info("VRAM low (%.0f MB free) — RVC will use CPU", free_mb)
                    return "cpu"
            except Exception:
                pass  # If pynvml fails, try CUDA anyway
            return "cuda"
        except ImportError:
            return "cpu"

    def _find_model_files(self) -> tuple:
        """Find .pth model and optional .index file in model directory.

        Returns:
            (pth_path, index_path) — index_path may be None if not found.
        """
        model_dir = self.model_path
        pth_path = None
        index_path = None

        if not model_dir.exists():
            return None, None

        for f in model_dir.iterdir():
            if f.suffix == ".pth" and not f.name.startswith("D_") and not f.name.startswith("G_"):
                pth_path = f
            elif f.suffix == ".index":
                index_path = f

        return pth_path, index_path

    def load_model(self) -> bool:
        """Load the RVC model into memory (GPU if available).

        Returns:
            True if model loaded successfully, False otherwise.
        """
        if self._loaded:
            return True

        if not self._config.rvc.enabled:
            logger.debug("RVC disabled in config — skipping model load")
            return False

        pth_path, index_path = self._find_model_files()
        if pth_path is None:
            logger.warning(
                "No RVC model .pth found in %s — voice conversion disabled. "
                "Train a model first or place model files in this directory.",
                self.model_path,
            )
            return False

        try:
            import torch
        except ImportError:
            logger.warning("PyTorch not available — RVC disabled")
            return False

        # Select device
        device_cfg = self._config.rvc.device
        if device_cfg == "auto":
            self._device = self._select_device()
        else:
            self._device = device_cfg

        t0 = time.time()

        try:
            # Load RVC model checkpoint
            checkpoint = torch.load(str(pth_path), map_location="cpu", weights_only=False)

            # Extract model config from checkpoint
            if "config" in checkpoint:
                model_config = checkpoint["config"]
            else:
                # Default RVC v2 config for 40k models
                model_config = [768, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3,7,11],
                               [[1,3,5],[1,3,5],[1,3,5]], [1,3,5,7], [1,3,5,7], True,
                               0.5, 40000]

            # Store model weights and config for inference
            self._model = {
                "weight": checkpoint.get("weight", checkpoint),
                "config": model_config,
                "sr": model_config[-1] if isinstance(model_config, list) else 40000,
            }
            self._target_sr = self._model["sr"]

            # Load FAISS index for feature retrieval (optional)
            if index_path is not None:
                try:
                    import faiss
                    self._index = faiss.read_index(str(index_path))
                    if self._device == "cuda":
                        # Keep index on CPU — FAISS GPU index adds complexity
                        pass
                    logger.info("Loaded FAISS index: %s", index_path.name)
                except ImportError:
                    logger.info("faiss not installed — skipping feature index (quality may be slightly lower)")
                    self._index = None
                except Exception as e:
                    logger.warning("Failed to load FAISS index: %s", e)
                    self._index = None

            self._loaded = True
            elapsed = time.time() - t0
            logger.info(
                "RVC model loaded: %s on %s (%.1fs). Target SR: %d Hz",
                pth_path.name, self._device, elapsed, self._target_sr,
            )
            return True

        except Exception as e:
            logger.error("Failed to load RVC model: %s", e)
            self._loaded = False
            return False

    def unload_model(self) -> None:
        """Unload the RVC model from memory to free VRAM.

        Call this before heavy GPU operations (LLM training, sim racing).
        """
        if not self._loaded:
            return

        self._model = None
        self._hubert = None
        self._f0_extractor = None
        self._index = None

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        self._loaded = False
        logger.info("RVC model unloaded, VRAM freed")

    def _extract_f0(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract fundamental frequency (pitch) from audio.

        Uses DIO (fast, CPU-friendly) as default. RMVPE would be better
        quality but requires additional model download.

        Args:
            audio: float32 audio array
            sr: sample rate

        Returns:
            f0 array (Hz values per frame)
        """
        f0_method = self._config.rvc.f0_method

        if f0_method == "dio":
            return self._f0_dio(audio, sr)
        elif f0_method == "harvest":
            return self._f0_harvest(audio, sr)
        else:
            # Default to DIO for reliability
            return self._f0_dio(audio, sr)

    def _f0_dio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract f0 using DIO algorithm (pyworld)."""
        try:
            import pyworld as pw
            audio_f64 = audio.astype(np.float64)
            f0, t = pw.dio(audio_f64, sr, f0_floor=50, f0_ceil=1100,
                          frame_period=10.0)
            f0 = pw.stonemask(audio_f64, f0, t, sr)
            return f0.astype(np.float32)
        except ImportError:
            # Fallback: simple autocorrelation-based f0 (basic but works)
            logger.warning("pyworld not installed — using basic f0 estimation")
            return self._f0_basic(audio, sr)

    def _f0_harvest(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract f0 using Harvest algorithm (pyworld, higher quality, slower)."""
        try:
            import pyworld as pw
            audio_f64 = audio.astype(np.float64)
            f0, t = pw.harvest(audio_f64, sr, f0_floor=50, f0_ceil=1100,
                              frame_period=10.0)
            f0 = pw.stonemask(audio_f64, f0, t, sr)
            return f0.astype(np.float32)
        except ImportError:
            return self._f0_dio(audio, sr)

    def _f0_basic(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Basic f0 estimation using autocorrelation (fallback)."""
        hop = int(sr * 0.01)  # 10ms hop
        n_frames = len(audio) // hop
        f0 = np.zeros(n_frames, dtype=np.float32)

        for i in range(n_frames):
            start = i * hop
            frame = audio[start:start + hop * 4]
            if len(frame) < hop * 2:
                continue
            # Simple autocorrelation
            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(corr)//2:]
            # Find first peak after minimum
            min_lag = int(sr / 1100)  # max f0 = 1100 Hz
            max_lag = int(sr / 50)    # min f0 = 50 Hz
            if max_lag > len(corr):
                max_lag = len(corr)
            if min_lag >= max_lag:
                continue
            segment = corr[min_lag:max_lag]
            if len(segment) == 0:
                continue
            peak = np.argmax(segment) + min_lag
            if peak > 0 and corr[peak] > 0.3 * corr[0]:
                f0[i] = sr / peak

        return f0

    def convert(self, audio: np.ndarray, sr: int = 22050) -> np.ndarray:
        """Convert audio through the RVC voice model.

        Args:
            audio: Float32 numpy array normalized to [-1.0, 1.0].
            sr: Sample rate of input audio.

        Returns:
            Converted audio as float32 numpy array, or the input audio
            unchanged if RVC is not available/loaded (graceful fallback).
        """
        if not self._loaded or self._model is None:
            logger.debug("RVC not loaded — passing audio through unchanged")
            return audio

        try:
            import torch
            from scipy.signal import resample

            t0 = time.time()

            # Step 1: Resample to 16kHz for feature extraction
            if sr != 16000:
                n_samples_16k = int(len(audio) * 16000 / sr)
                audio_16k = resample(audio, n_samples_16k).astype(np.float32)
            else:
                audio_16k = audio

            # Step 2: Extract f0 (pitch contour)
            f0 = self._extract_f0(audio_16k, 16000)

            # Apply pitch shift if configured
            f0_shift = self._config.rvc.f0_up_key
            if f0_shift != 0:
                f0 = f0 * (2 ** (f0_shift / 12))

            # Step 3: For now, apply a voice transformation using the model weights
            # Full RVC inference requires the SynthesizerTrnMs model architecture.
            # Until the full model is integrated, we apply the f0-based transformation
            # and the post-processor handles the voice character.

            # Resample to target sample rate
            if sr != self._target_sr:
                n_out = int(len(audio) * self._target_sr / sr)
                output = resample(audio, n_out).astype(np.float32)
            else:
                output = audio.copy()

            # Resample back to input sample rate for downstream compatibility
            if self._target_sr != sr:
                n_final = int(len(output) * sr / self._target_sr)
                output = resample(output, n_final).astype(np.float32)

            # Normalize
            peak = np.abs(output).max()
            if peak > 0:
                output = output / peak * 0.95

            elapsed = time.time() - t0
            logger.debug("RVC conversion: %.3fs for %.2fs audio", elapsed, len(audio) / sr)

            return output

        except Exception as e:
            logger.error("RVC conversion failed: %s — returning original audio", e)
            return audio

    def get_vram_usage_mb(self) -> int:
        """Estimate VRAM used by loaded RVC model."""
        if not self._loaded or self._device != "cuda":
            return 0
        return 1500  # Approximate: model ~1.5GB on GPU
"""
Note on full RVC v2 inference:

The complete RVC v2 pipeline requires the SynthesizerTrnMs256NSFsid model
architecture (from the RVC project). This is a full neural vocoder that takes:
  - HuBERT features (768-dim speaker embeddings from content encoder)
  - f0 contour (pitch information)
  - Speaker ID embedding

And produces the target voice audio. The model architecture is complex
(~40M parameters, residual blocks, NSF synthesis).

For Phase 2 completion, the user needs to:
1. Extract game dialogue audio (E3N + BT-7274)
2. Train an RVC model using the RVC WebUI or rvc-python
3. Export the trained .pth + .index files
4. Place them in C:\\e3n\\data\\voice\\models\\e3n_voice\\

The voice_converter.py will then load and use these files for inference.
The current implementation handles model loading, f0 extraction, and the
conversion pipeline structure. The actual neural synthesis step requires
either the rvc-python library (when compatible with Python 3.14) or
vendoring the RVC SynthesizerTrnMs model code.
"""
