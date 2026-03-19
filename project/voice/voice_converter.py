"""
RVC voice conversion — STUB for Phase 2.

Will convert TTS output through a trained RVC model to clone E3N's target voice.
For now, passes audio through unchanged as a graceful fallback.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .voice_config import VoiceConfig, DEFAULT_CONFIG

logger = logging.getLogger("e3n.voice.rvc")


class VoiceConverter:
    """RVC-based voice conversion.

    Phase 2 implementation will:
    - Load a trained RVC model from C:\\e3n\\data\\voice\\models\\
    - Convert TTS audio to match a target voice timbre
    - Manage VRAM (load/unload for GPU sharing with LLM inference)

    Current behavior: passthrough (returns input audio unchanged).
    """

    def __init__(self, config: Optional[VoiceConfig] = None) -> None:
        self._config = config or DEFAULT_CONFIG
        self._model = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Whether the RVC model is currently loaded in memory."""
        return self._loaded

    @property
    def model_path(self) -> Path:
        """Path to the RVC model directory."""
        return Path(self._config.rvc.model_dir) / self._config.rvc.model_name

    def load_model(self) -> bool:
        """Load the RVC model into memory (GPU if available).

        Returns:
            True if model loaded successfully, False otherwise.

        Note:
            Phase 2 — currently a no-op stub.
        """
        if not self._config.rvc.enabled:
            logger.debug("RVC disabled in config — skipping model load")
            return False

        model_dir = self.model_path
        if not model_dir.exists():
            logger.warning(
                "RVC model not found at %s — voice conversion disabled. "
                "Train a model with train_rvc.py first.",
                model_dir,
            )
            return False

        # Phase 2: actual RVC model loading goes here
        # - Load .pth model weights
        # - Load .index file for feature matching
        # - Initialize f0 extractor (rmvpe/crepe)
        # - Move to GPU if VRAM available
        logger.info("RVC model loading not yet implemented (Phase 2)")
        return False

    def unload_model(self) -> None:
        """Unload the RVC model from memory to free VRAM.

        Call this before LLM inference if VRAM is contested.

        Note:
            Phase 2 — currently a no-op stub.
        """
        if self._loaded:
            # Phase 2: release model tensors, clear CUDA cache
            self._model = None
            self._loaded = False
            logger.info("RVC model unloaded")

    def convert(self, audio: np.ndarray) -> np.ndarray:
        """Convert audio through the RVC voice model.

        Args:
            audio: Float32 numpy array normalized to [-1.0, 1.0].

        Returns:
            Converted audio as float32 numpy array, or the input audio
            unchanged if RVC is not available/loaded (graceful fallback).
        """
        if not self._loaded or self._model is None:
            # Graceful passthrough — no conversion applied
            logger.debug("RVC not loaded — passing audio through unchanged")
            return audio

        # Phase 2: actual voice conversion pipeline
        # 1. Extract f0 (pitch) using configured method
        # 2. Extract speaker embedding
        # 3. Run RVC inference
        # 4. Return converted audio
        return audio
