"""
Piper TTS engine wrapper.

Synthesizes text to 16-bit PCM audio using Piper (if installed).
Model loaded lazily on first call. Graceful fallback if Piper unavailable.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .voice_config import VoiceConfig, DEFAULT_CONFIG

logger = logging.getLogger("e3n.voice.tts")


class PiperTTS:
    """Text-to-speech engine using Piper.

    Lazily loads the voice model on first synthesis call.
    Returns audio as float32 numpy array normalized to [-1.0, 1.0].
    """

    def __init__(self, config: Optional[VoiceConfig] = None) -> None:
        self._config = config or DEFAULT_CONFIG
        self._model = None
        self._available: Optional[bool] = None

    @property
    def sample_rate(self) -> int:
        """Output sample rate in Hz."""
        return self._config.tts.sample_rate

    @property
    def is_available(self) -> bool:
        """Check if Piper TTS is importable."""
        if self._available is None:
            try:
                import piper  # noqa: F401
                self._available = True
            except ImportError:
                self._available = False
                logger.warning(
                    "Piper TTS not installed — TTS synthesis disabled. "
                    "Install with: pip install piper-tts"
                )
        return self._available

    def _load_model(self) -> bool:
        """Lazily load the Piper voice model.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        if self._model is not None:
            return True

        if not self.is_available:
            return False

        model_dir = Path(self._config.tts.model_dir)
        model_name = self._config.tts.model_name

        # Piper expects an .onnx model file
        model_path = model_dir / f"{model_name}.onnx"

        if not model_path.exists():
            logger.warning(
                "Piper model not found at %s — TTS disabled. "
                "Download a voice model from https://github.com/rhasspy/piper",
                model_path,
            )
            return False

        try:
            from piper import PiperVoice

            self._model = PiperVoice.load(str(model_path))
            logger.info("Piper TTS model loaded: %s", model_name)
            return True
        except Exception as e:
            logger.error("Failed to load Piper model '%s': %s", model_name, e)
            self._model = None
            return False

    def synthesize(self, text: str) -> Optional[np.ndarray]:
        """Synthesize text to audio.

        Args:
            text: Input text to speak.

        Returns:
            Float32 numpy array normalized to [-1.0, 1.0] at self.sample_rate,
            or None if TTS is unavailable or synthesis fails.
        """
        if not text or not text.strip():
            logger.debug("Empty text passed to synthesize — skipping")
            return None

        if not self._load_model():
            return None

        try:
            import io
            import wave

            # Piper synthesizes to a WAV stream
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                self._model.synthesize(
                    text,
                    wav_file,
                    speaker_id=self._config.tts.speaker_id,
                )

            # Read raw PCM from the WAV buffer
            wav_buffer.seek(0)
            with wave.open(wav_buffer, "rb") as wav_file:
                n_frames = wav_file.getnframes()
                raw_bytes = wav_file.readframes(n_frames)
                sample_width = wav_file.getsampwidth()

            # Convert to float32 normalized [-1.0, 1.0]
            if sample_width == 2:
                pcm = np.frombuffer(raw_bytes, dtype=np.int16)
                audio = pcm.astype(np.float32) / 32768.0
            elif sample_width == 4:
                pcm = np.frombuffer(raw_bytes, dtype=np.int32)
                audio = pcm.astype(np.float32) / 2147483648.0
            else:
                pcm = np.frombuffer(raw_bytes, dtype=np.uint8)
                audio = (pcm.astype(np.float32) - 128.0) / 128.0

            logger.debug(
                "Synthesized %d samples (%.2fs) for text: '%.40s...'",
                len(audio),
                len(audio) / self.sample_rate,
                text,
            )
            return audio

        except Exception as e:
            logger.error("TTS synthesis failed: %s", e)
            return None

    def unload(self) -> None:
        """Release the loaded model from memory."""
        if self._model is not None:
            self._model = None
            logger.info("Piper TTS model unloaded")
