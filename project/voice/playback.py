"""
Audio playback with priority queue.

Non-blocking async playback using sounddevice (with graceful fallback).
High-priority audio interrupts current playback.
"""

import asyncio
import logging

import numpy as np

from .voice_config import DEFAULT_CONFIG, VoiceConfig

logger = logging.getLogger("deltai.voice.playback")

# Check for sounddevice
try:
    import sounddevice as sd

    _HAS_SOUNDDEVICE = True
except ImportError:
    sd = None
    _HAS_SOUNDDEVICE = False
    logger.warning(
        "sounddevice not installed — audio playback disabled. Install with: pip install sounddevice"
    )


# Priority levels
PRIORITY_LOW = 0
PRIORITY_NORMAL = 1
PRIORITY_HIGH = 2

_PRIORITY_MAP = {
    "low": PRIORITY_LOW,
    "normal": PRIORITY_NORMAL,
    "high": PRIORITY_HIGH,
}


class Playback:
    """Async audio playback with priority queue.

    High-priority audio interrupts current playback.
    Graceful fallback if sounddevice is not installed or no audio device available.
    """

    def __init__(self, config: VoiceConfig | None = None) -> None:
        self._config = config or DEFAULT_CONFIG
        self._current_priority: int = PRIORITY_LOW
        self._playing: bool = False
        self._stop_event: asyncio.Event | None = None
        self._available: bool | None = None

    @property
    def is_available(self) -> bool:
        """Check if audio playback is functional."""
        if self._available is None:
            if not _HAS_SOUNDDEVICE:
                self._available = False
            else:
                try:
                    devices = sd.query_devices()
                    # Check for at least one output device
                    self._available = any(
                        d.get("max_output_channels", 0) > 0
                        for d in (devices if isinstance(devices, list) else [devices])
                    )
                    if not self._available:
                        logger.warning("No audio output devices found — playback disabled")
                except Exception as e:
                    logger.warning("Failed to query audio devices: %s — playback disabled", e)
                    self._available = False
        return self._available

    @property
    def is_playing(self) -> bool:
        """Whether audio is currently playing."""
        return self._playing

    async def play(
        self,
        audio: np.ndarray,
        sample_rate: int = 22050,
        priority: str = "normal",
    ) -> bool:
        """Play audio with priority handling.

        Args:
            audio: Float32 numpy array normalized to [-1.0, 1.0].
            sample_rate: Audio sample rate in Hz.
            priority: "low", "normal", or "high". High interrupts current playback.

        Returns:
            True if playback completed or was queued, False if unavailable.
        """
        if audio is None or len(audio) == 0:
            return False

        if not self.is_available:
            logger.debug("Playback unavailable — audio discarded")
            return False

        pri = _PRIORITY_MAP.get(priority, PRIORITY_NORMAL)

        # If currently playing, check priority
        if self._playing:
            if pri > self._current_priority:
                logger.debug("High-priority audio interrupting current playback")
                self.stop()
            else:
                logger.debug(
                    "Audio queued (priority %s <= current %s) — dropping",
                    priority,
                    self._current_priority,
                )
                return False

        self._playing = True
        self._current_priority = pri
        self._stop_event = asyncio.Event()

        try:
            await self._play_async(audio, sample_rate)
            return True
        except Exception:
            logger.error("Playback failed")
            return False
        finally:
            self._playing = False
            self._current_priority = PRIORITY_LOW

    async def _play_async(self, audio: np.ndarray, sample_rate: int) -> None:
        """Non-blocking playback via sounddevice in a thread executor."""
        loop = asyncio.get_event_loop()
        cfg = self._config.playback

        def _blocking_play() -> None:
            try:
                sd.play(
                    audio,
                    samplerate=sample_rate,
                    device=cfg.device_index,
                    blocksize=cfg.blocksize,
                )
                sd.wait()
            except Exception as e:
                logger.error("sounddevice playback error: %s", e)

        await loop.run_in_executor(None, _blocking_play)

    def stop(self) -> None:
        """Stop current playback immediately."""
        if not _HAS_SOUNDDEVICE:
            return
        try:
            sd.stop()
            self._playing = False
            logger.debug("Playback stopped")
        except Exception as e:
            logger.warning("Error stopping playback: %s", e)
