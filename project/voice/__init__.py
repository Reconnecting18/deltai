"""
E3N Voice Module — TTS -> RVC -> PostProcess -> Playback pipeline.

Usage:
    from voice import speak
    await speak("Copy that, Ethan.")
    await speak("Pit window opens in 2 laps.", priority="high")

Pipeline:
    1. PiperTTS synthesizes text to raw audio
    2. VoiceConverter applies RVC voice cloning (Phase 2 — passthrough for now)
    3. PostProcessor applies electronic effects chain
    4. Playback outputs audio to speakers with priority handling

Every stage degrades gracefully: if Piper isn't installed, speak() logs a
warning and returns. If RVC isn't loaded, audio passes through unchanged.
If no audio device exists, audio is silently discarded.
"""

import asyncio
import logging
from typing import Optional

import numpy as np

from .voice_config import VoiceConfig, DEFAULT_CONFIG
from .tts_engine import PiperTTS
from .voice_converter import VoiceConverter
from .post_processor import PostProcessor
from .playback import Playback

logger = logging.getLogger("e3n.voice")

__all__ = [
    "speak",
    "VoiceConfig",
    "DEFAULT_CONFIG",
    "PiperTTS",
    "VoiceConverter",
    "PostProcessor",
    "Playback",
]

# Module-level pipeline components (lazy-initialized)
_tts: Optional[PiperTTS] = None
_rvc: Optional[VoiceConverter] = None
_post: Optional[PostProcessor] = None
_player: Optional[Playback] = None
_config: VoiceConfig = DEFAULT_CONFIG


def configure(config: Optional[VoiceConfig] = None) -> None:
    """Reconfigure the voice pipeline with new settings.

    Reinitializes all pipeline components. Call before speak()
    if you need non-default settings.

    Args:
        config: VoiceConfig to use. If None, resets to DEFAULT_CONFIG.
    """
    global _tts, _rvc, _post, _player, _config
    _config = config or DEFAULT_CONFIG
    _tts = None
    _rvc = None
    _post = None
    _player = None
    logger.info("Voice pipeline reconfigured")


def _ensure_components() -> None:
    """Lazy-initialize pipeline components on first use."""
    global _tts, _rvc, _post, _player
    if _tts is None:
        _tts = PiperTTS(_config)
    if _rvc is None:
        _rvc = VoiceConverter(_config)
    if _post is None:
        _post = PostProcessor(_config)
    if _player is None:
        _player = Playback(_config)


async def speak(text: str, priority: str = "normal") -> bool:
    """Speak text through the full voice pipeline.

    Orchestrates: TTS -> RVC (if available) -> PostProcess -> Playback.

    Args:
        text: Text to synthesize and speak.
        priority: "low", "normal", or "high". High-priority audio
                  interrupts current playback.

    Returns:
        True if audio was played (or at least attempted), False if
        the pipeline could not produce audio (e.g., TTS unavailable).
    """
    if not text or not text.strip():
        return False

    _ensure_components()

    # Step 1: TTS — text to raw audio
    audio = _tts.synthesize(text)
    if audio is None:
        logger.warning("TTS produced no audio for: '%.50s...' — pipeline aborted", text)
        return False

    sample_rate = _tts.sample_rate

    # Step 2: RVC voice conversion (passthrough if not loaded)
    audio = _rvc.convert(audio)

    # Step 3: Post-processing effects chain
    audio = _post.process(audio, sample_rate)

    # Step 4: Playback
    played = await _player.play(audio, sample_rate, priority)

    if played:
        logger.debug("Spoke: '%.50s...' (priority=%s)", text, priority)
    else:
        logger.debug("Playback skipped for: '%.50s...'", text)

    return played


def synthesize_only(text: str) -> Optional[np.ndarray]:
    """Run TTS + RVC + PostProcess but skip playback.

    Useful for getting the processed audio buffer (e.g., to encode
    as WAV/MP3 for a web response).

    Args:
        text: Text to synthesize.

    Returns:
        Float32 numpy array of processed audio, or None if TTS failed.
    """
    if not text or not text.strip():
        return None

    _ensure_components()

    audio = _tts.synthesize(text)
    if audio is None:
        return None

    audio = _rvc.convert(audio)
    audio = _post.process(audio, _tts.sample_rate)

    return audio


# ── BACKWARD COMPATIBILITY — re-export legacy voice functions ──────────
# The original voice.py (now voice_legacy.py) contains the existing STT/TTS
# pipeline (Whisper + edge-tts). These are re-exported here so all existing
# imports (`from voice import synthesize_speech, transcribe_audio, ...`) still work.
try:
    from voice_legacy import (  # noqa: F401
        synthesize_speech,
        transcribe_audio,
        get_voice_status,
        record_audio,
        VOICE_ENABLED,
        _init_whisper,
        _clean_for_tts,
        _windows_tts,
        _check_tts,
    )
except ImportError:
    VOICE_ENABLED = False  # Legacy voice module not available
